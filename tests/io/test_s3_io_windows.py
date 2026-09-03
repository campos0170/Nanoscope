"""Third-sweep loader defects, part 2: a window means the rows it says it means.

  IO3-04  A windowed CSV load counted PHYSICAL LINES where the whole-file load counts DATA
          ROWS, because ``skiprows=range(1, 1 + start)`` is applied by pandas before its own
          ``skip_blank_lines=True``. One blank line after the header — what Excel and several
          instrument exporters write — slid the window one row early, with ``meta.issues``
          empty and ``meta.loaded_range`` still naming the range that was asked for. Adjacent
          windows of one file then stopped tiling: a row duplicated at every seam, the tail
          rows dropped.

  IO3-11  ``SampleRange(0, 0)`` on a CSV fell through to ``CorruptFileError`` "no data rows
          were read" — corruption wording, naming neither the request nor the file's real
          length, for an intact recording — because the sweep-2 zero-overlap reroute was gated
          on ``start > 0``. The identical request to HDF5 already answered "the requested
          sample range [0, 0) selects nothing in a 100-sample recording".

Both are the same rule: **what a load returns, what its metadata says it returned, and what
the file holds are one description.**
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, per tests/conftest.py
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import csv as csv_module
from pathlib import Path

import h5py
import numpy as np
import pytest

from nanoscope.core.types import SampleRange
from nanoscope.io.base import open_document
from nanoscope.io.errors import ReaderError, UnsupportedFormatError

# Sample values equal their DATA-ROW index, so a slid window is readable straight off the data
# and no oracle is needed to see it.
_N_ROWS = 5_000
_DT = 1e-4


def _rows(n: int = _N_ROWS):
    return [(i * _DT, float(i)) for i in range(n)]


def _write_csv(path: Path, *, blank_every: int | None = None,
               blank_after_header: bool = False, newline: str = "\n") -> Path:
    out = ["time,current"]
    if blank_after_header:
        out.append("")
    for i, (t, v) in enumerate(_rows()):
        out.append(f"{t!r},{v!r}")
        if blank_every and (i + 1) % blank_every == 0:
            out.append("")
    path.write_bytes((newline.join(out) + newline).encode("utf-8"))
    return path


def _stdlib_truth(path: Path) -> list[list[str]]:
    """The file's data rows as the stdlib sees them — no pandas, no nanoscope."""
    with path.open(newline="") as fh:
        rows = [r for r in csv_module.reader(fh) if r and any(f.strip() for f in r)]
    return rows[1:]


# =================================================================================================
# IO3-04: blank lines shift physical lines, not data rows
# =================================================================================================

@pytest.mark.parametrize("label,kwargs", [
    ("no blanks (control)", {}),
    ("one blank after the header", {"blank_after_header": True}),
    ("a blank every 500 rows", {"blank_every": 500}),
    ("CRLF and a blank after the header", {"blank_after_header": True, "newline": "\r\n"}),
])
def test_a_blank_line_before_the_window_does_not_slide_it(tmp_path, label, kwargs):
    """[MEASURED] before: window[0] = 1999.0 for a request starting at 2,000 with one blank
    line, 1997.0 with a blank every 500 rows, while loaded_range still said (2000, 2100)."""
    p = _write_csv(tmp_path / "w.csv", **kwargs)
    whole = open_document(p)
    window = open_document(p, SampleRange(2_000, 2_100))

    assert whole.meta.n_samples == _N_ROWS, f"{label}: blanks are not data rows"
    assert window.channels[0][0] == pytest.approx(whole.channels[0][2_000]), (
        f"{label}: sample 2,000 of the file and sample 0 of the window are the same row")
    assert window.meta.loaded_range == SampleRange(2_000, 2_100), label
    assert window.meta.issues == (), f"{label}: an ordinary file gained an issue: {window.meta.issues!r}"


def test_the_windows_clock_is_the_row_it_really_starts_at(tmp_path):
    """The other half of the slide, and the one nothing on screen would ever show: t0_s was
    computed from the ASKED start, so the window's clock was wrong by the blank count too."""
    p = _write_csv(tmp_path / "clock.csv", blank_every=500)
    truth = _stdlib_truth(p)
    doc = open_document(p, SampleRange(2_000, 2_100))
    assert doc.meta.t0_s == pytest.approx(float(truth[2_000][0]), abs=1e-12)


def test_adjacent_windows_of_a_file_with_blank_lines_tile_it_exactly(tmp_path):
    """The consequence that is not visible in any single load: windows are tiled by sample
    index, so a slid window duplicates a row at every seam and loses the file's tail. Before
    the fix this reconstruction differed from the file in 3,000 of 4,000 samples."""
    p = _write_csv(tmp_path / "tile.csv", blank_every=500)
    truth = [float(r[1]) for r in _stdlib_truth(p)]

    rebuilt: list[float] = []
    for a in range(0, _N_ROWS, 1_000):
        rebuilt.extend(open_document(p, SampleRange(a, a + 1_000)).channels[0].tolist())
    assert len(rebuilt) == len(truth)
    assert rebuilt == pytest.approx(truth), "tiled windows do not reconstruct the file"


def test_a_window_that_runs_past_the_end_still_reports_what_it_holds(tmp_path):
    """The boundary the fix must not move: a partial overlap keeps loading what exists and
    keeps saying so."""
    p = _write_csv(tmp_path / "short.csv", blank_after_header=True)
    doc = open_document(p, SampleRange(_N_ROWS - 50, _N_ROWS + 950))
    assert doc.meta.n_samples == 50
    assert doc.meta.loaded_range == SampleRange(_N_ROWS - 50, _N_ROWS)
    assert any("Nothing is wrong with the file" in s for s in doc.meta.issues), doc.meta.issues


# =================================================================================================
# IO3-11: an empty request is a request problem at start == 0 too
# =================================================================================================

def _short_csv(tmp_path: Path, rows: int = 100) -> Path:
    p = tmp_path / "trace.csv"
    with p.open("w") as f:
        f.write("time,current\n")
        for i in range(rows):
            f.write(f"{i / 1000.0:.6f},{0.1 * i:.3f}\n")
    return p


def _short_h5(tmp_path: Path, rows: int = 100) -> Path:
    p = tmp_path / "trace.h5"
    with h5py.File(p, "w") as h:
        h["current"] = np.zeros(rows)
        h["time"] = np.arange(rows) / 1000.0
    return p


def test_an_empty_request_on_a_csv_says_selects_nothing_not_corruption(tmp_path):
    """Was: CorruptFileError "trace.csv: no data rows were read." for an intact 100-row CSV."""
    with pytest.raises(UnsupportedFormatError) as exc:
        open_document(_short_csv(tmp_path), SampleRange(0, 0))
    message = str(exc.value)
    assert "selects nothing" in message, message
    assert "[0, 0)" in message, "the refusal must name the request"
    assert "100-sample recording" in message, "the refusal must name the file's real length"
    assert "no data rows were read" not in message


def test_the_csv_and_hdf5_empty_request_sentences_are_identical(tmp_path):
    """The defect was the disagreement, so pin the two together rather than the wording."""
    csv = _short_csv(tmp_path)
    h5 = _short_h5(tmp_path)

    def refusal(path: Path) -> str:
        with pytest.raises(ReaderError) as exc:
            open_document(path, SampleRange(0, 0))
        return str(exc.value).removeprefix(path.name)

    assert refusal(csv) == refusal(h5)


def test_a_real_window_at_the_top_of_the_file_still_loads(tmp_path):
    """No false positive: start == 0 with a real stop is an ordinary window."""
    doc = open_document(_short_csv(tmp_path), SampleRange(0, 10))
    assert doc.meta.n_samples == 10
    assert doc.meta.loaded_range == SampleRange(0, 10)
    assert doc.meta.t0_s == pytest.approx(0.0)
