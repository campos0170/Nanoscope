"""EXP2-01, unit level — ``StagedSet.unstage`` and ``save_figure``'s use of it.

``StagedSet.stage`` reserves a member BEFORE the caller writes it, so a write that
fails after staging left an orphan; ``commit()`` then raised FileNotFoundError over the
never-written temp and the WHOLE set — including families that were fully written —
was abandoned. ``unstage`` is the withdrawal door: the failing family removes exactly
its own members and the survivors still commit. ``save_figure`` withdraws its whole
family on any failure with a shared group (a partial withdrawal would commit a PDF
whose PNG/CSV are a previous export's — the mixed vintage the set exists to prevent).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from nanoscope.core.atomic_write import StagedSet


def _write(tmp: Path, payload: bytes = b"x") -> None:
    tmp.write_bytes(payload)


def test_unstage_removes_exactly_the_named_members(tmp_path):
    group = StagedSet()
    a1, a2 = tmp_path / "a.pdf", tmp_path / "a.png"
    b1, b2 = tmp_path / "b.pdf", tmp_path / "b.png"
    _write(group.stage(a1), b"A1")
    _write(group.stage(a2), b"A2")
    tmp_b1 = group.stage(b1)
    _write(tmp_b1, b"B1")
    tmp_b2 = group.stage(b2)          # reserved, never written — the EXP2-01 orphan

    group.unstage([b1, b2])

    assert set(group.targets) == {a1, a2}
    assert not tmp_b1.exists() and not tmp_b2.exists(), \
        "unstage left the withdrawn family's temps on disk"

    replaced = group.commit()          # must NOT raise over the withdrawn orphan
    assert set(replaced) == {a1, a2}
    assert a1.read_bytes() == b"A1" and a2.read_bytes() == b"A2"
    assert not b1.exists() and not b2.exists()


def test_unstage_matches_through_a_symlinked_directory(tmp_path):
    """``stage`` records ``os.path.realpath(target)`` (resolve_write_target), so a member
    staged through a symlinked DIRECTORY lives in the set under its real path. ``unstage``
    must resolve the caller's spelling the same way; comparing the bare spelling matched
    nothing, the never-written temp stayed staged, and ``commit`` refused the WHOLE set —
    the healthy-figures-sunk failure this fix exists to end. Not exotic on this platform:
    ``/tmp`` → ``/private/tmp`` and ``/var`` → ``/private/var`` on every Mac, and any
    export folder the user reaches through a Finder alias/symlink has the same shape."""
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir, target_is_directory=True)

    group = StagedSet()
    keep = link_dir / "keep.pdf"
    drop = link_dir / "drop.pdf"
    _write(group.stage(keep), b"KEEP")
    tmp_drop = group.stage(drop)       # reserved, never written — the orphan

    group.unstage([drop])              # named by the SAME symlinked spelling

    assert not tmp_drop.exists(), "the withdrawn member's temp stayed on disk"
    assert [t.name for t in group.targets] == ["keep.pdf"], (
        "unstage silently withdrew nothing for a symlinked-dir target: "
        f"{[str(t) for t in group.targets]}")

    replaced = group.commit()          # must NOT refuse over the withdrawn orphan
    assert [t.name for t in replaced] == ["keep.pdf"]
    assert (real_dir / "keep.pdf").read_bytes() == b"KEEP"
    assert not (real_dir / "drop.pdf").exists()


def test_save_figure_failure_in_a_symlinked_dir_still_frees_the_shared_group(tmp_path):
    """End-to-end through the exact wiring ``_export_figures`` uses, with the export
    directory reached through a symlink (as any ``/tmp``- or ``/var``-spelled path is on
    macOS): the poisoned figure's whole family is withdrawn and the healthy family still
    commits — no FileNotFoundError blaming the environment, no partial poison family."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import pandas as pd
    from matplotlib.figure import Figure

    from nanoscope.plot.figures import save_figure

    poisoned = Figure(figsize=(2, 2))
    ax = poisoned.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    ax.set_title(r"$\badmathcommand{x}$")
    try:
        poisoned.savefig(io.BytesIO(), format="pdf")
    except Exception:
        pass
    else:
        pytest.skip("this matplotlib renders the poison mathtext; the shape is gone")

    real_dir = tmp_path / "export"
    real_dir.mkdir()
    link_dir = tmp_path / "alias"
    link_dir.symlink_to(real_dir, target_is_directory=True)

    healthy = Figure(figsize=(2, 2))
    healthy.add_subplot(111).plot([0, 1], [1, 0])
    df = pd.DataFrame({"x": [0, 1]})

    group = StagedSet()
    save_figure(healthy, link_dir / "figures_good.pdf", data=df, group=group)
    with pytest.raises(Exception):
        save_figure(poisoned, link_dir / "figures_bad.pdf", data=df, group=group)

    group.commit()                     # the healthy family still lands
    names = sorted(p.name for p in real_dir.iterdir())
    assert names == ["figures_good.csv", "figures_good.pdf", "figures_good.png"], names


def test_commit_without_unstage_still_refuses_over_an_orphan(tmp_path):
    """The guard the withdrawal must not weaken: an orphan NOBODY withdrew still
    refuses the commit — that is the all-or-nothing promise."""
    group = StagedSet()
    _write(group.stage(tmp_path / "a.pdf"))
    group.stage(tmp_path / "b.pdf")    # never written, never withdrawn
    with pytest.raises(FileNotFoundError):
        group.commit()
    assert not (tmp_path / "a.pdf").exists(), "the refusal must leave nothing behind"


def test_save_figure_withdraws_its_whole_family_from_a_shared_group(tmp_path):
    """The figure builds, ``savefig`` raises at draw time (bad mathtext): with a shared
    group, save_figure must leave the group exactly as it found it — the PDF member it
    staged before the failure included — so the caller's other figures still commit."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import pandas as pd
    from matplotlib.figure import Figure

    from nanoscope.plot.figures import save_figure

    poisoned = Figure(figsize=(2, 2))
    ax = poisoned.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    ax.set_title(r"$\badmathcommand{x}$")
    try:
        poisoned.savefig(io.BytesIO(), format="pdf")
    except Exception:
        pass
    else:
        pytest.skip("this matplotlib renders the poison mathtext; the shape is gone")

    healthy = Figure(figsize=(2, 2))
    healthy.add_subplot(111).plot([0, 1], [1, 0])
    df = pd.DataFrame({"x": [0, 1]})

    group = StagedSet()
    out = save_figure(healthy, tmp_path / "figures_good.pdf", data=df, group=group)
    n_before = len(group.targets)

    with pytest.raises(Exception):
        save_figure(poisoned, tmp_path / "figures_bad.pdf", data=df, group=group)

    assert len(group.targets) == n_before, (
        "save_figure left its failed family staged in the shared group: "
        f"{[str(t) for t in group.targets]}")
    assert not any("figures_bad" in t.name for t in group.targets)

    replaced = group.commit()          # the healthy family still lands
    assert (tmp_path / "figures_good.pdf").exists()
    assert (tmp_path / "figures_good.png").exists()
    assert not list(tmp_path.glob("figures_bad.*")), list(tmp_path.iterdir())
    assert set(replaced) == set(Path(p) for p in out.values())


def test_save_figure_alone_still_abandons_its_private_group(tmp_path):
    """The ``group is None`` leg is not this fix's to change: a lone save that fails
    still leaves nothing on disk."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import pandas as pd
    from matplotlib.figure import Figure

    from nanoscope.plot.figures import save_figure

    poisoned = Figure(figsize=(2, 2))
    ax = poisoned.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    ax.set_title(r"$\badmathcommand{x}$")
    try:
        poisoned.savefig(io.BytesIO(), format="pdf")
    except Exception:
        pass
    else:
        pytest.skip("this matplotlib renders the poison mathtext; the shape is gone")

    with pytest.raises(Exception):
        save_figure(poisoned, tmp_path / "figures_solo.pdf",
                    data=pd.DataFrame({"x": [0, 1]}))
    assert list(tmp_path.iterdir()) == [], list(tmp_path.iterdir())
