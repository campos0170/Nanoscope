"""CR5-02 / CR5-03 — a DAMAGED project is refused as damaged, through one exception type.

* **CR5-02.** ``zipfile.is_zipfile`` checks the end-of-central-directory — the exact bytes
  an interrupted copy destroys — so every truncated genuine project was refused as
  "not a NanoScope project (not a ZIP archive)": a wrong-file diagnosis for a file that IS
  the user's project, steering them away from the honest remedies (re-copy, restore a
  backup). Measured: losing the last 200 bytes of a 912 KB project — 99.98% intact, still
  leading with the ZIP magic — got the wrong-file sentence. The gate now reads the LEADING
  bytes: ``PK\\x03\\x04`` means "was an archive; is damaged", and the refusal says so.
* **CR5-03.** ``load_project`` documents exactly one exception family (ProjectError, a
  ValueError), and ``_manifest_section`` folds manifest malformations into it — but
  IN-PLACE corruption (a CRC-damaged member) surfaced from ``zf.read`` as
  ``zipfile.BadZipFile``, which subclasses neither. An API caller catching ProjectError
  per the docs crashed; the GUI survived only via its blanket handler and printed raw zip
  internals. The whole read now lives inside a ``BadZipFile``/``zlib.error`` boundary.

The control matters: a file that never was a ZIP must KEEP the wrong-file sentence —
"damaged" for a text file would send its owner restoring backups of a file that was never
a project.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanoscope.core.project import (
    ProjectError,
    ProjectState,
    ResultRec,
    load_project,
    save_project,
)


def _events_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "event_id": np.arange(n),
        "start_idx": np.arange(n) * 500,
        "end_idx": np.arange(n) * 500 + 120,
        "depth": rng.normal(30.0, 2.0, n),
        "accepted": np.ones(n, dtype=bool),
    })


@pytest.fixture
def project(tmp_path) -> Path:
    """A genuine project written by the app's own save, big enough to truncate inside."""
    path = tmp_path / "study.nanoscope"
    state = ProjectState(results=[
        ResultRec("/data/pore_a.csv", 0, "detect.schmitt", _events_df(),
                  provenance={"module": "detect.schmitt"})])
    save_project(path, state)
    assert load_project(path).results, "the fixture project must round-trip before damage"
    return path


# ================================================================================================
# CR5-02 — truncation is named as damage, not as a wrong file
# ================================================================================================

@pytest.mark.parametrize("keep", [0.25, 0.75, 0.9998])
def test_a_truncated_project_is_refused_as_damaged(project, keep):
    raw = project.read_bytes()
    cut = project.with_name("study_copy.nanoscope")
    cut.write_bytes(raw[: max(4, int(len(raw) * keep))])
    with pytest.raises(ProjectError) as err:
        load_project(cut)
    msg = str(err.value)
    assert "damaged or truncated" in msg, msg
    assert "backup" in msg, "the refusal must point at the honest remedy"
    assert "is not a NanoScope project" not in msg, (
        f"the wrong-file diagnosis must not appear for a damaged project: {msg}")


def test_a_file_that_never_was_a_zip_keeps_the_wrong_file_sentence(tmp_path):
    junk = tmp_path / "not_a_zip.nanoscope"
    junk.write_text("this is not a zip archive")
    with pytest.raises(ProjectError, match="not a NanoScope project"):
        load_project(junk)
    with pytest.raises(ProjectError) as err:
        load_project(junk)
    assert "damaged or truncated" not in str(err.value)


def test_a_tiny_stub_without_the_magic_is_not_called_damaged(tmp_path):
    stub = tmp_path / "stub.nanoscope"
    stub.write_bytes(b"PK")   # too short to carry the local-header magic
    with pytest.raises(ProjectError, match="not a NanoScope project"):
        load_project(stub)


# ================================================================================================
# CR5-03 — in-place corruption folds into ProjectError
# ================================================================================================

def _flip_inside_member(path: Path, member: str) -> None:
    """XOR 64 bytes in the middle of ``member``'s compressed data — what bit-rot does."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(member)
    raw = bytearray(path.read_bytes())
    # Local header is 30 bytes + name; land safely inside the compressed payload.
    start = info.header_offset + 30 + len(member) + info.compress_size // 2
    for i in range(start, min(start + 64, len(raw))):
        raw[i] ^= 0xFF
    path.write_bytes(bytes(raw))


def test_crc_damage_raises_project_error_not_badzipfile(project):
    _flip_inside_member(project, "results/0.parquet")
    assert zipfile.is_zipfile(project), "in-place damage leaves the tail intact"
    with pytest.raises(ProjectError) as err:
        load_project(project)
    msg = str(err.value)
    assert "damaged" in msg and "integrity" in msg, msg
    assert "backup" in msg, "the refusal must point at the honest remedy"


def test_the_documented_contract_holds_for_crc_damage(project):
    """The API promise verbatim: one family, and it is a ValueError."""
    _flip_inside_member(project, "results/0.parquet")
    try:
        load_project(project)
    except ValueError as exc:                 # ProjectError subclasses ValueError
        assert isinstance(exc, ProjectError)
        assert exc.__cause__ is not None, "the raw BadZipFile stays reachable for debugging"
    else:
        pytest.fail("a CRC-damaged member must refuse, not load")


def test_an_intact_project_still_loads(project):
    """CONTROL: neither gate may tax the healthy path."""
    state = load_project(project)
    assert len(state.results) == 1
    assert len(state.results[0].events) == 40
