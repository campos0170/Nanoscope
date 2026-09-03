"""The text reader takes channel KIND from an explicit column name (units stay unknown), so a CSV
with a `current` column is analysable. Regression for: detection greyed out on CSVs because their
current column was labelled UNKNOWN (2026-07-18)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import nanoscope  # noqa: F401
import pytest

from nanoscope.core.types import ChannelKind
from nanoscope.io.base import open_document, probe_path


def _write_csv(path, header):
    lines = [header]
    for i in range(2000):
        t = i / 50000.0
        cols = header.count(",") + 1
        row = [f"{t:.6f}"] + ["-50.0"] * (cols - 1)
        lines.append(",".join(row))
    path.write_text("\n".join(lines))
    return path


def test_current_column_is_kind_current(tmp_path):
    p = _write_csv(tmp_path / "trace.csv", "time,current")
    meta = probe_path(p).meta
    cur = [c for c in meta.channels if c.kind is ChannelKind.CURRENT]
    assert len(cur) == 1 and cur[0].name == "current"
    assert cur[0].raw_units == "", "units must stay uninferred even when kind is named"


def test_named_columns_map_to_kinds(tmp_path):
    p = _write_csv(tmp_path / "multi.csv", "time,current,voltage,conductance")
    kinds = {c.name: c.kind for c in probe_path(p).meta.channels}
    assert kinds["current"] is ChannelKind.CURRENT
    assert kinds["voltage"] is ChannelKind.VOLTAGE
    assert kinds["conductance"] is ChannelKind.CONDUCTANCE


def test_unrecognized_column_stays_unknown(tmp_path):
    p = _write_csv(tmp_path / "mystery.csv", "time,signal_a")
    (info,) = probe_path(p).meta.channels
    assert info.kind is ChannelKind.UNKNOWN, "arbitrary names must not be mislabelled"


@pytest.mark.corpus
def test_real_csv_current_is_detectable(tmp_path):
    from pathlib import Path
    f = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.csv")
    if not f.exists():
        pytest.skip("corpus CSV not present")
    doc = open_document(f)
    cur = [c for c in doc.meta.channels if c.kind is ChannelKind.CURRENT]
    assert cur, "real CSV current column must be recognised as CURRENT so detection can run"
