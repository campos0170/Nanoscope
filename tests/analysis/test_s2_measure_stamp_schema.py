"""MEAS2-01 / MEAS2-02, data layer: a saved row carries its calibration + clock stamps.

The manual-measurement schema used to strip the ``units`` echo as a diagnostic and had
no calibration or clock columns at all, so rows measured under gain 1 (30 pA), gain 2
(60 pA) and a pA→nA change (0.03 nA) pooled into one table — and one exported CSV —
with nothing to tell them apart, while ``t_start_s`` said nothing about which clock it
was on. These tests pin the pure layer: the four provenance columns
(``units``/``cal_gain``/``cal_offset``/``t0_s``), their additive back-compat defaults,
and :func:`calibration_stamp_state`'s three-way answer (match / stale / unknown —
"a fingerprint that cannot be compared must read as 'unknown', never as 'matches'").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.measure import (
    CAL_STAMP_MATCH,
    CAL_STAMP_STALE,
    CAL_STAMP_UNKNOWN,
    MEASUREMENT_DIAGNOSTIC_KEYS,
    MEASUREMENT_META_COLUMNS,
    MEASUREMENT_SCHEMA,
    append_measurement,
    calibration_stamp_state,
    coerce_measurements,
    empty_measurements_df,
    measure_selection,
    schema_default,
    write_measurements_csv,
)

FS = 100_000.0

_STAMP_COLUMNS = ("units", "cal_gain", "cal_offset", "t0_s")


def _square(n=20_000, i0=100.0, level=70.0, start=8_000, stop=12_000) -> np.ndarray:
    x = np.full(n, i0, dtype=np.float32)
    x[start:stop] = level
    return x


def _measured_row(**stamps) -> dict:
    """One measure_selection dict with caller stamps, like the panel builds it."""
    m = measure_selection(_square(), FS, 8_000, 12_000, units=stamps.pop("units", "pA"))
    for key in MEASUREMENT_DIAGNOSTIC_KEYS:
        m.pop(key, None)
    m.update({"recording": "r.dat", "doc_key": "/tmp/r.dat", "channel": "current",
              "channel_index": 0, "label": ""})
    m.update(stamps)
    return m


# ================================================================================================
# the schema carries the stamps
# ================================================================================================

def test_schema_has_the_provenance_columns():
    for col in _STAMP_COLUMNS:
        assert col in MEASUREMENT_SCHEMA, col
        # Caller-stamped provenance, not a computed statistic.
        assert col in MEASUREMENT_META_COLUMNS or col == "units", col
    # The units echo is no longer a stripped diagnostic — it is what tells a pA row
    # from an nA row in the pooled table.
    assert "units" not in MEASUREMENT_DIAGNOSTIC_KEYS
    assert "units" in MEASUREMENT_META_COLUMNS


def test_append_keeps_units_and_calibration_stamp():
    row = _measured_row(units="pA", cal_gain=2.0, cal_offset=0.5, t0_s=61.3034)
    df = append_measurement(empty_measurements_df(), row)
    got = df.iloc[0]
    assert str(got["units"]) == "pA"
    assert float(got["cal_gain"]) == 2.0
    assert float(got["cal_offset"]) == 0.5
    assert float(got["t0_s"]) == pytest.approx(61.3034)


def test_a_row_with_no_stamps_defaults_to_unknown_not_zero():
    """A caller that stamps nothing (or ``None``, the no-clock case) must land on the
    documented "not recorded" defaults — NaN / '' — never on a fabricated 1.0/0.0 that
    would read as a real identity calibration."""
    row = _measured_row(units="")
    row["t0_s"] = None                 # clock_offset_s(None-rate meta) hands back None
    df = append_measurement(empty_measurements_df(), row)
    got = df.iloc[0]
    assert np.isnan(float(got["cal_gain"]))
    assert np.isnan(float(got["cal_offset"]))
    assert np.isnan(float(got["t0_s"]))
    assert schema_default("cal_gain") != schema_default("cal_gain")   # NaN, definitionally


def test_legacy_frame_coerces_with_unknown_stamps(tmp_path):
    """A frame written before the stamp columns existed (an old project / CSV) loads
    with '' / NaN — the additive back-compat contract — and still round-trips CSV."""
    legacy = append_measurement(
        empty_measurements_df(), _measured_row()).drop(columns=list(_STAMP_COLUMNS))
    back = coerce_measurements(legacy)
    assert list(back.columns) == list(MEASUREMENT_SCHEMA)
    assert str(back.iloc[0]["units"]) == ""
    assert np.isnan(float(back.iloc[0]["cal_gain"]))
    assert np.isnan(float(back.iloc[0]["t0_s"]))

    out = write_measurements_csv(legacy, tmp_path / "legacy.csv")
    header = out.read_text().splitlines()[0].split(",")
    for col in _STAMP_COLUMNS:
        assert col in header, col


def test_csv_export_writes_the_stamp_columns(tmp_path):
    df = append_measurement(
        empty_measurements_df(),
        _measured_row(units="nA", cal_gain=0.001, cal_offset=0.0, t0_s=1154.99))
    out = write_measurements_csv(df, tmp_path / "m.csv")
    text = out.read_text()
    header = text.splitlines()[0].split(",")
    for col in _STAMP_COLUMNS:
        assert col in header, col
    row = pd.read_csv(out).iloc[0]
    assert row["units"] == "nA"
    assert float(row["cal_gain"]) == 0.001
    assert float(row["t0_s"]) == pytest.approx(1154.99)


# ================================================================================================
# calibration_stamp_state: match / stale / unknown
# ================================================================================================

def test_stamp_state_match_and_stale_on_gain():
    assert calibration_stamp_state(1.0, 0.0, "pA", 1.0, 0.0, "pA") == CAL_STAMP_MATCH
    assert calibration_stamp_state(1.0, 0.0, "pA", 2.0, 0.0, "pA") == CAL_STAMP_STALE
    assert calibration_stamp_state(1.0, 0.5, "pA", 1.0, 0.0, "pA") == CAL_STAMP_STALE


def test_stamp_state_units_change_alone_is_stale():
    """A pA→nA relabel with the same gain still changes what every signal number MEANS."""
    assert calibration_stamp_state(1.0, 0.0, "pA", 1.0, 0.0, "nA") == CAL_STAMP_STALE


def test_stamp_state_unknown_never_reads_as_match_or_stale():
    """The pre-stamp row (NaN gain/offset) can be compared with nothing: claiming
    'stale' would assert knowledge nobody has, claiming 'match' would be worse."""
    nan = float("nan")
    for live in ((1.0, 0.0, "pA"), (2.0, 1.0, "nA")):
        assert calibration_stamp_state(nan, nan, "", *live) == CAL_STAMP_UNKNOWN
        assert calibration_stamp_state(None, None, None, *live) == CAL_STAMP_UNKNOWN


def test_stamp_state_handles_na_units_from_a_hand_edited_csv():
    """A units cell that arrives as NaN (a hole in a hand-edited CSV) must compare as
    'no label', not as the string 'nan'."""
    assert calibration_stamp_state(1.0, 0.0, float("nan"), 1.0, 0.0, "") == CAL_STAMP_MATCH
