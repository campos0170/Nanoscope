"""C13 — a project save that fails must cost the user nothing.

``zipfile.ZipFile(path, "w")`` truncates the target the instant it opens, so the pre-fix
``save_project`` destroyed the existing project BEFORE it knew whether it could write a new
one. Injecting ENOSPC on the second parquet member turned a good 6949-byte project into a
3251-byte corpse that then failed to load at all — and the GUI's "Could not save the project"
box appeared after the file was already gone.

Every test here is written against the FAILURE, not against the happy path: the assertion is
always "the file that was there before is still exactly there".
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.project import (
    DocumentRef,
    ProjectState,
    ResultRec,
    load_project,
    save_project,
)


# ================================================================================================
# builders
# ================================================================================================
def _state(n_results: int = 3, tag: str = "a") -> ProjectState:
    return ProjectState(
        documents=[DocumentRef(path="/data/rec.abf", loaded_range=None)],
        results=[
            ResultRec(
                doc_id="/data/rec.abf", channel=0, module_id=f"{tag}{i}",
                events=pd.DataFrame({"start_idx": np.arange(64) + i,
                                     "accepted": np.ones(64, dtype=bool)}),
            )
            for i in range(n_results)
        ],
    )


def _fail_on_nth_parquet(monkeypatch, n: int, exc: BaseException) -> None:
    """Make the n-th ``results/*.parquet`` member raise — a disk filling up mid-write."""
    real = zipfile.ZipFile.writestr
    seen = {"n": 0}

    def boom(self, name, data, *a, **kw):
        if str(getattr(name, "filename", name)).endswith(".parquet"):
            seen["n"] += 1
            if seen["n"] == n:
                raise exc
        return real(self, name, data, *a, **kw)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", boom)


def _leftovers(directory: Path, target: Path) -> list[Path]:
    return [p for p in directory.iterdir() if p != target]


# ================================================================================================
# THE regression: a failed save leaves the previous project byte-for-byte intact
# ================================================================================================
def test_a_failed_save_does_not_destroy_the_existing_project(tmp_path, monkeypatch):
    """C13, exactly as reproduced: ENOSPC on the SECOND parquet member.

    Pre-fix this shrank the file (6949 -> 3251 bytes in the original repro) and the reload
    raised ``ProjectError: ... is missing its manifest.json``.
    """
    target = tmp_path / "study.nanoscope"
    save_project(target, _state(3, tag="good"))
    before = target.read_bytes()
    assert len(before) > 0

    _fail_on_nth_parquet(monkeypatch, 2, OSError(28, "No space left on device"))
    with pytest.raises(OSError):
        save_project(target, _state(3, tag="new"))

    assert target.read_bytes() == before, "the previous project was modified by a failed save"
    restored = load_project(target)                       # and it is still READABLE
    assert [r.module_id for r in restored.results] == ["good0", "good1", "good2"]


def test_a_failed_save_leaves_no_temporary_behind(tmp_path, monkeypatch):
    """Atomicity must not be bought by littering the user's project folder."""
    target = tmp_path / "study.nanoscope"
    save_project(target, _state(2))

    _fail_on_nth_parquet(monkeypatch, 1, OSError(28, "No space left on device"))
    with pytest.raises(OSError):
        save_project(target, _state(2))

    assert _leftovers(tmp_path, target) == []


def test_a_successful_save_leaves_no_temporary_behind(tmp_path):
    target = tmp_path / "study.nanoscope"
    save_project(target, _state(2))
    save_project(target, _state(4))
    assert _leftovers(tmp_path, target) == []
    assert len(load_project(target).results) == 4


def test_a_failed_FIRST_save_creates_no_file_at_all(tmp_path, monkeypatch):
    """Nothing was there; nothing must be left. A half-written new project reads as a
    corrupt one, and the user has no way to know it never finished."""
    target = tmp_path / "fresh.nanoscope"
    _fail_on_nth_parquet(monkeypatch, 1, OSError(28, "No space left on device"))
    with pytest.raises(OSError):
        save_project(target, _state(2))
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_non_OSError_failure_is_equally_survivable(tmp_path, monkeypatch):
    """The guarantee is about ANY failure, not about disk-full specifically.

    An unserialisable frame raises from ``to_parquet`` deep inside the write; that must
    protect the existing project exactly as ENOSPC does.
    """
    target = tmp_path / "study.nanoscope"
    save_project(target, _state(3, tag="good"))
    before = target.read_bytes()

    bad = _state(2)
    bad.results[1].events = pd.DataFrame({"x": [object()]})   # pyarrow cannot serialise this
    with pytest.raises(Exception):
        save_project(target, bad)

    assert target.read_bytes() == before
    assert _leftovers(tmp_path, target) == []


def test_a_keyboard_interrupt_mid_save_is_survivable(tmp_path, monkeypatch):
    """Ctrl-C is a BaseException, not an Exception. A bare ``except Exception`` around the
    staging would let it escape with the temp already renamed — or, pre-fix, with the
    target already truncated."""
    target = tmp_path / "study.nanoscope"
    save_project(target, _state(3, tag="good"))
    before = target.read_bytes()

    _fail_on_nth_parquet(monkeypatch, 2, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        save_project(target, _state(3))

    assert target.read_bytes() == before
    assert _leftovers(tmp_path, target) == []


def test_the_replace_is_atomic_not_a_truncate_and_copy(tmp_path):
    """The target must never be observed empty or partial.

    A save that truncates and refills has a window where the file exists but is not a
    project. os.replace has no such window — the old inode stays whole until the instant
    the new one takes its name. We prove it by holding the OLD file open across the save:
    an in-place rewrite would corrupt what this handle sees; a rename cannot.
    """
    target = tmp_path / "study.nanoscope"
    save_project(target, _state(3, tag="old"))
    before = target.read_bytes()

    with open(target, "rb") as held:
        head = held.read(4)
        save_project(target, _state(5, tag="new"))      # replaces the NAME, not the inode
        rest = held.read()

    assert head == b"PK\x03\x04"
    # The held descriptor still sees the ORIGINAL archive, whole. An in-place rewrite would
    # have replaced the bytes under this very handle.
    assert head + rest == before
    assert len(load_project(target).results) == 5       # ...and the name now has the new one


def test_the_saved_project_keeps_the_permissions_it_had(tmp_path):
    """os.replace installs a NEW inode. Without carrying the mode over, a project the user
    had locked down to 0600 would silently come back at whatever the umask allows."""
    target = tmp_path / "study.nanoscope"
    save_project(target, _state(1))
    os.chmod(target, 0o600)
    save_project(target, _state(2))
    assert os.stat(target).st_mode & 0o777 == 0o600


def test_save_still_returns_the_path_and_round_trips(tmp_path):
    """The atomic rewrite must not have changed the function's contract."""
    target = tmp_path / "study.nanoscope"
    out = save_project(target, _state(2))
    assert out == target
    state = load_project(out)
    assert len(state.results) == 2
    assert state.documents[0].path == "/data/rec.abf"
