"""The unit of a CSV time column: read from the header when it names one, asked about when
it does not and the rate reads too slow to be seconds.

[MEASURED] before the fix, through ``probe_path`` on this tree: a ``time_ms,current_pA`` export
at 10 µs steps (0.01 per row) reported 100 Hz, ``Provenance.MEASURED``, duration 552 s for 0.5 s
of samples, ``issues=()`` — the header said milliseconds in so many letters and nothing read it,
and the only guard (the 1 Hz floor, tests/io/test_s3_io_clocks.py IO3-05) sees microseconds
but not milliseconds. These pin the three tiers of ``io/base.py`` on both doors.
"""
import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, per tests/conftest.py
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import h5py
import numpy as np
import pytest

from nanoscope.core.types import Provenance, SampleRange
from nanoscope.io.base import SUSPECT_UNDECLARED_RATE_HZ, open_document, probe_path
from nanoscope.io.text import _time_unit_from_header

N = 6_000
STEP_S = 1e-5            # 100 kHz


def _text(strings) -> str:
    return " || ".join(strings)


def _write_csv(path: Path, header: str, times, values=None) -> Path:
    values = np.zeros(len(times)) if values is None else values
    rows = [header] + [f"{t!r},{v!r}" for t, v in zip(times, values)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


# =================================================================================================
# tier 1: a header that names the unit is a declaration, and the column is read in it
# =================================================================================================

@pytest.mark.parametrize("header, per_unit", [
    ("time_ms,current", 1e-3),
    ("Time (ms),Current (pA)", 1e-3),
    ("t_us,i", 1e-6),
    ("t [µs],current", 1e-6),          # the micro sign, as instrument software writes it
    ("time [ns],current", 1e-9),
    ("time_s,current", 1.0),
    ("Time (sec),current", 1.0),
])
def test_a_declared_unit_is_read_as_declared_on_both_doors(tmp_path, header, per_unit):
    times = [i * STEP_S / per_unit for i in range(N)]       # the same 100 kHz clock, in that unit
    p = _write_csv(tmp_path / "declared.csv", header, times)
    for label, meta in (("probe", probe_path(p).meta), ("load", open_document(p).meta)):
        assert meta.sample_rate_hz == pytest.approx(1.0 / STEP_S, rel=1e-9), label
        assert meta.rate_provenance is Provenance.MEASURED, label
        assert meta.time_scale_s == per_unit, label
        assert meta.issues == (), f"{label}: a declared unit is not a problem: {meta.issues!r}"
        said = _text(meta.notes)
        assert "declares" in said and "in its header" in said, f"{label}: {meta.notes!r}"
        assert header.split(",")[0].strip() in said, "the note names the column it read"


def test_a_declared_unit_scales_the_start_time_and_the_window_clock(tmp_path):
    """t0 is read out of the column (USER_GUIDE §2), so it scales with the column — for the
    whole file and for a window, whose t0 is the column's own value at that row."""
    offset_ms = 2_500.0                                           # the recording starts at 2.5 s
    times = [offset_ms + i * 0.01 for i in range(N)]
    p = _write_csv(tmp_path / "offset_ms.csv", "time_ms,current", times)
    whole = open_document(p).meta
    assert whole.t0_s == pytest.approx(2.5)
    assert whole.duration_s == pytest.approx(N * STEP_S)
    window = open_document(p, SampleRange(1_000, 2_000)).meta
    assert window.t0_s == pytest.approx(2.5 + 1_000 * STEP_S)
    assert window.sample_rate_hz == pytest.approx(1.0 / STEP_S, rel=1e-9)
    assert window.issues == (), "the per-sample clock audit must agree with the scaled rate"


def test_a_declared_microsecond_column_is_no_longer_refused(tmp_path):
    """The twin of IO3-05's CSV case with the one difference that matters: the header says
    what the unit is, so the floor (which exists for UNDECLARED columns) does not apply."""
    p = _write_csv(tmp_path / "declared_us.csv", "time_us,current",
                   [float(i * 10) for i in range(N)])
    meta = probe_path(p).meta
    assert meta.sample_rate_hz == pytest.approx(100_000.0)
    assert meta.rate_provenance is Provenance.MEASURED
    assert "will not guess" not in _text(meta.issues)


def test_a_refused_hdf5_time_base_keeps_its_raw_t0_so_a_declared_unit_recovers_it(tmp_path):
    """The HDF5 twin of the CSV rule: at the refusal the /time value at the window's start is
    kept AS WRITTEN (2,500,000 µs), inert while the rate is None, so that 'the unit is µs'
    resolves rate AND t0 — for the whole file (2.5 s) and for a window (2.51 s), the
    start/fs apart that contract.window_t0 exists to keep."""
    from nanoscope.core.sample_rate import SampleRateDeclaration
    from nanoscope.core.session import Session
    with h5py.File(tmp_path / "us.h5", "w") as h:
        h["current"] = np.zeros(N)
        h["time"] = (2_500_000 + np.arange(N) * 10).astype(np.int64)
    whole, window = open_document(tmp_path / "us.h5"), open_document(tmp_path / "us.h5",
                                                                        SampleRange(1_000, 2_000))
    assert whole.meta.sample_rate_hz is None and whole.meta.refused_rate_hz == pytest.approx(0.1)
    assert whole.meta.t0_s == 2_500_000.0 and window.meta.t0_s == 2_510_000.0
    session = Session()
    us = SampleRateDeclaration("time_unit", time_unit="us")
    for doc, expect in ((whole, 2.5), (window, 2.51)):
        key = session.add_document(doc)
        session.set_sample_rate_declaration(key, us)
        meta = session.document(key).meta
        assert meta.sample_rate_hz == pytest.approx(100_000.0)
        assert meta.t0_s == pytest.approx(expect)


@pytest.mark.parametrize("header", ["time", "timestamp", "items", "time_ticks", "ms_s"])
def test_these_headers_declare_nothing(header):
    """A bare ``time``; a word that merely CONTAINS a unit token (``items`` holds ``ms``, and
    substring matching would have read it); an unknown token; two units at once."""
    assert _time_unit_from_header(header) is None


# =================================================================================================
# tier 2 (unchanged): below one sample per second, an undeclared column is refused
# =================================================================================================

def test_an_undeclared_column_below_one_hertz_is_still_refused_but_its_interval_is_kept(tmp_path):
    """The refusal is unchanged (rate None, UNKNOWN, the floor's sentence); what is new is
    that the interval it measured travels on the meta — refused_rate_hz — so the unit can be
    declared in File Info, which is the remedy the sentence now names. Both doors, and the
    HDF5 twin of the same refusal."""
    p = _write_csv(tmp_path / "bare_us.csv", "time,current", [float(i * 100) for i in range(N)])
    for label, meta in (("probe", probe_path(p).meta), ("load", open_document(p).meta)):
        assert meta.sample_rate_hz is None and meta.rate_provenance is Provenance.UNKNOWN, label
        assert meta.refused_rate_hz == pytest.approx(0.01), label
        said = _text(meta.issues)
        assert "less than one sample per second" in said and "will not guess" in said
        assert "declare the unit of this time base" in said, "the remedy the floor names"
    with h5py.File(tmp_path / "int_us.h5", "w") as h:
        h["current"] = np.zeros(N)
        h["time"] = (np.arange(N) * 100).astype(np.int64)
    meta = probe_path(tmp_path / "int_us.h5").meta
    assert meta.sample_rate_hz is None and meta.refused_rate_hz == pytest.approx(0.01)
    assert "the column" not in _text(meta.issues), "/time is a dataset, not a column"
    ok = _write_csv(tmp_path / "ok.csv", "time,current", [i * STEP_S for i in range(N)])
    assert probe_path(ok).meta.refused_rate_hz is None, "nothing refused: nothing kept"


# =================================================================================================
# tier 3: below a kilohertz, an undeclared column is asked about — and its rate is KEPT
# =================================================================================================

def test_an_undeclared_millisecond_column_is_asked_about_not_loaded_in_silence(tmp_path):
    """The defect. 0.01 per row read as seconds is 100 Hz, which the floor accepts."""
    p = _write_csv(tmp_path / "bare_ms.csv", "time,current", [i * 0.01 for i in range(N)])
    for label, meta in (("probe", probe_path(p).meta), ("load", open_document(p).meta)):
        assert meta.sample_rate_hz == pytest.approx(100.0), f"{label}: the rate is kept as read"
        assert meta.rate_provenance is Provenance.MEASURED, label
        said = _text(meta.issues)
        assert "100 Hz" in said and "not declared" in said, f"{label}: {meta.issues!r}"
        assert "milliseconds" in said and "100,000 Hz" in said, "it says what ms would mean"
        assert "kept as read" in said, "it says that nothing was converted"
        assert "time_ms" in said and "File Info" in said, "it names both remedies"
        assert "will not guess" not in said, "this is not the refusal sentence"
    assert probe_path(p).meta.issues == open_document(p).meta.issues, "one sentence, both doors"


def test_the_bound_is_a_kilohertz_and_a_recording_exactly_there_gains_nothing(tmp_path):
    """The 1 kHz fixtures elsewhere in this suite are written as ``i / 1000.0``; their measured
    rate is 1 kHz up to the rounding of their own decimals and must not fall under the bound."""
    assert SUSPECT_UNDECLARED_RATE_HZ == 1000.0
    at = _write_csv(tmp_path / "1khz.csv", "time,current", [i / 1000.0 for i in range(N)])
    under = _write_csv(tmp_path / "999hz.csv", "time,current", [i / 999.0 for i in range(N)])
    assert probe_path(at).meta.issues == ()
    assert "not declared" in _text(probe_path(under).meta.issues)


@pytest.mark.parametrize("dt", [1e-4, 2e-5])
def test_ordinary_undeclared_recordings_still_gain_nothing(tmp_path, dt):
    p = _write_csv(tmp_path / "ok.csv", "time,current", [i * dt for i in range(N)])
    assert probe_path(p).meta.issues == ()
    assert open_document(p).meta.issues == ()


def test_the_same_question_is_asked_of_an_hdf5_time_dataset(tmp_path):
    """/time is as undeclared as a CSV column; the reader door asks one question in one
    sentence for both formats."""
    with h5py.File(tmp_path / "ms.h5", "w") as h:
        h["current"] = np.zeros(N)
        h["time"] = np.arange(N) * 0.01                      # milliseconds, as float seconds
    meta = probe_path(tmp_path / "ms.h5").meta
    assert meta.sample_rate_hz == pytest.approx(100.0)
    said = _text(meta.issues)
    assert "/time" in said and "not declared" in said and "kept as read" in said
    assert "time_ms" not in said, "the header remedy is a CSV's; HDF5 has no header to name"
    assert "the column" not in said, "/time is a dataset; the sentence must not call it a column"
