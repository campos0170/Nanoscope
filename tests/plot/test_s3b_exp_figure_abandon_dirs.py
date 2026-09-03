"""EXP3-03 — an abandoned figure export must not leave the folder tree it created.

``StagedSet`` was given ``_made_dirs`` and the cleanup in :meth:`abandon` for exactly
one reason, in its own words: "'Nothing on disk changes' is the promise; an empty folder
named after an export that never happened is a change, and a confusing one -- the user
goes looking in it for the twelve files the status line said were not written."

The guard could never fire for the figure export, because ``save_figure`` created the
directory ITSELF one line before it staged anything::

    stem.parent.mkdir(parents=True, exist_ok=True)

By the time ``StagedSet.stage`` walked upward looking for missing parents there were
none, ``_made_dirs`` stayed empty, and an export into ``paper/figures/v3`` that failed on
every figure left all three levels behind — under a dialog that says "Nothing was written
— every file already in the folder is exactly as it was."

The fix is a removal: the directory is now created by ``stage``, which is the thing that
knows how to undo it. These tests pin both halves — the tree goes when the export is
abandoned, and it STAYS when the export succeeds (it holds the export now).
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from nanoscope.core.atomic_write import StagedSet
from nanoscope.plot.figures import save_figure


@pytest.fixture
def frame():
    return pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})


def _plain_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


def _poison_figure():
    """A figure that renders fine until ``savefig`` reaches its title.

    Same trigger the app already documents for this path: invalid mathtext carried
    into a title from the data file. It raises at DRAW time, i.e. inside the staged
    write, which is the only way to reach the abandon path without faking anything.
    """
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title(r"$\badmathcommand{x}$")
    return fig


def _poison_really_fails() -> bool:
    """Renders into memory, so the probe leaves nothing in the folder under test."""
    fig = _poison_figure()
    try:
        fig.savefig(io.BytesIO(), format="pdf")
        return False
    except Exception:
        return True
    finally:
        plt.close(fig)


# ================================================================================================
# the defect
# ================================================================================================

def test_a_figure_export_that_fails_removes_every_level_it_created(tmp_path, frame):
    """The GUI shape: ONE shared group across the whole export, every figure failing,
    then ``group.abandon()`` — exactly what ``MainWindow._export_figures`` does when
    nothing was written."""
    if not _poison_really_fails():
        pytest.skip("this matplotlib renders the poisoned title; the trigger is gone")

    deep = tmp_path / "paper" / "figures" / "v3"
    group = StagedSet()
    fig = _poison_figure()
    try:
        with pytest.raises(Exception):
            save_figure(fig, deep / "figures_dwell.pdf", data=frame, group=group)
    finally:
        plt.close(fig)

    # The export is over and nothing was written; the caller abandons the set.
    group.abandon()

    assert not deep.exists(), (
        "the abandoned export left the folder it created behind — the dialog says "
        "'Nothing was written', so an empty paper/figures/v3 is a change the user is "
        "not told about")
    assert not (tmp_path / "paper" / "figures").exists()
    assert not (tmp_path / "paper").exists()
    assert list(tmp_path.iterdir()) == [], "nothing at all should be left in the folder"


def test_save_figure_on_its_own_group_also_undoes_the_tree(tmp_path, frame):
    """``group=None`` — every non-GUI caller — abandons its own set on failure and must
    clean up the same way."""
    if not _poison_really_fails():
        pytest.skip("this matplotlib renders the poisoned title; the trigger is gone")

    deep = tmp_path / "solo" / "nested"
    fig = _poison_figure()
    try:
        with pytest.raises(Exception):
            save_figure(fig, deep / "figures_solo.pdf", data=frame)
    finally:
        plt.close(fig)

    assert not deep.exists() and not (tmp_path / "solo").exists()


# ================================================================================================
# ...and the other half: a SUCCESSFUL export keeps its folder
# ================================================================================================

def test_a_successful_export_keeps_the_folder_it_created(tmp_path, frame):
    """The removal must not become "the export deletes its own destination". A committed
    set drops ``_made_dirs`` deliberately — "they hold the export now; they are not ours
    to remove"."""
    deep = tmp_path / "paper" / "figures" / "v3"
    fig = _plain_figure()
    try:
        out = save_figure(fig, deep / "figures_dwell.pdf", data=frame)
    finally:
        plt.close(fig)

    assert deep.is_dir()
    for key in ("pdf", "png", "csv"):
        assert out[key].is_file() and out[key].stat().st_size > 0


def test_an_export_into_an_existing_folder_never_removes_it(tmp_path, frame):
    """``abandon`` may only remove what THIS set created. A folder that was already
    there — the user's own manuscript directory, with the previous export in it — must
    survive a failed export untouched."""
    if not _poison_really_fails():
        pytest.skip("this matplotlib renders the poisoned title; the trigger is gone")

    existing = tmp_path / "manuscript"
    existing.mkdir()
    keeper = existing / "figures_dwell.pdf"
    keeper.write_bytes(b"the previous export")

    fig = _poison_figure()
    try:
        with pytest.raises(Exception):
            save_figure(fig, keeper, data=frame)
    finally:
        plt.close(fig)

    assert existing.is_dir(), "an export never removes a folder it did not create"
    assert keeper.read_bytes() == b"the previous export"
