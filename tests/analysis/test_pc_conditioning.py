"""The conditioned trace as a pure kernel: the same array, the same bytes, exportable.

``analysis/conditioning.py`` exists because three surfaces need ONE processed trace and
only a private GUI staticmethod could build it. The whole value of the move is that the
new function is not merely *equivalent* to ``AnalysisPanel._conditioned_from_form`` — it
is bit-identical, so the curve the user is looking at, the numbers the Measure tool
reports and the samples the export writes are one array. These tests hold that:

  - identical BYTES (sha256) to the panel's own chain, on synthetic 50 kHz float32 and on
    the owner's real -1000 mV apo recording, under five chains including the empty one;
  - ``chain_description`` names every enabled stage, in the order the kernel applies them
    (a sentence that reordered them would describe a different trace);
  - the basis column is genuinely ADDITIVE: a frame that has never heard of it loads and
    gains ``''`` — "not recorded" — never "raw";
  - the export writes the samples, the time base and enough provenance that feeding the
    sidecar's own ``params`` back into ``conditioned_trace`` reproduces the array bit for
    bit; a slice says which slice it is.
"""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import (the panel cross-check imports gui). --------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import hashlib
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.conditioning import (
    BASIS_PROCESSED,
    BASIS_UNKNOWN,
    SIGNAL_BASIS_COLUMN,
    chain_description,
    conditioned_trace,
    conditioning_enabled,
    ensure_signal_basis_column,
    processed_trace_export_paths,
    write_processed_trace,
)

FS = 50_000.0
N = 100_000

#: The chains the owner actually configures, plus the empty one. Every stage that changes
#: the array is represented; `nostage` is here because "no chain" must return the input
#: untouched rather than a float32 copy that hashes differently.
CHAINS: dict[str, dict] = {
    "nostage": {},
    "lowpass only": {"filter_cutoff_hz": 5000.0, "filter_order": 8,
                     "filter_kind": "bessel"},
    "notch only": {"notch_enabled": True, "notch_freq_hz": 60.0, "notch_q": 30.0,
                   "notch_harmonics": 3},
    "detrend + lowpass": {"detrend_enabled": True, "detrend_method": "block_median",
                          "detrend_window_s": 0.05, "filter_cutoff_hz": 2000.0,
                          "filter_order": 4, "filter_kind": "butter"},
    "full chain": {"notch_enabled": True, "notch_freq_hz": 60.0, "notch_q": 30.0,
                   "notch_harmonics": 3, "detrend_enabled": True,
                   "detrend_method": "block_median", "detrend_window_s": 0.05,
                   "savgol_enabled": True, "savgol_window_s": 0.001, "savgol_order": 3,
                   "filter_cutoff_hz": 5000.0, "filter_order": 8,
                   "filter_kind": "bessel"},
}


def _trace(seed: int = 7) -> np.ndarray:
    """100 k float32 at 50 kHz: an open pore at 100 pA, mains hum, drift, one blockade."""
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=np.float64) / FS
    x = (100.0
         + 1.5 * rng.normal(size=N)
         + 2.0 * np.sin(2 * np.pi * 60.0 * t)      # something for the notch to remove
         + 3.0 * np.sin(2 * np.pi * 0.4 * t))      # something for the detrend to flatten
    x[40_000:44_000] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ================================================================================================
# 1. the same array, to the byte
# ================================================================================================
@pytest.mark.parametrize("name", list(CHAINS))
def test_conditioned_trace_is_bit_identical_to_the_panels_own_chain(name):
    """The pure kernel and ``AnalysisPanel._conditioned_from_form`` are ONE array.

    Not ``allclose``: the preview curve, the Measure readout and the export must be the
    same bytes, because two float32 traces that differ in the last bit are two traces
    nobody can tell apart by eye and every downstream statistic quietly disagrees about.
    """
    from nanoscope.gui.analysis_panel import AnalysisPanel

    raw, params = _trace(), CHAINS[name]
    mine = conditioned_trace(raw, FS, params)
    panel = AnalysisPanel._conditioned_from_form(raw, FS, params)
    assert _digest(mine) == _digest(panel)
    assert mine.dtype == panel.dtype == np.float32


@pytest.mark.corpus
@pytest.mark.parametrize("name", list(CHAINS))
def test_the_owners_own_recording_conditions_to_the_same_bytes_as_the_panel(name):
    """The same identity on the file this is FOR: 749,651 samples, ch2, 50 kHz.

    Measured when this landed (sha256, first 16 hex, mine == panel in every row):
    lowpass ``bcc676bf3fc86534``, notch ``0bf76e2566bcfad1``, detrend+lp
    ``5d4e225a92eb6588``, full ``e517e4381553e536``, nostage ``a62c3cb8a831db09``.
    """
    from nanoscope.gui.analysis_panel import AnalysisPanel
    from nanoscope.io.base import open_document

    p = pathlib.Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0004_"
                     "process _ -1000 mV_ 385-400 s time trace.abf")
    if not p.exists():
        pytest.skip(f"corpus file absent: {p.name}")
    doc = open_document(str(p))
    raw = np.asarray(doc.raw(2))
    fs = float(doc.meta.sample_rate_hz)
    params = CHAINS[name]
    assert _digest(conditioned_trace(raw, fs, params)) == \
        _digest(AnalysisPanel._conditioned_from_form(raw, fs, params))


def test_an_empty_chain_returns_the_samples_it_was_given():
    """No stage enabled -> the raw array, unchanged. The identity case matters because it
    is what "measure on: processed" must NOT be offered for."""
    raw = _trace()
    out = conditioned_trace(raw, FS, {})
    assert _digest(out) == _digest(raw)
    assert conditioning_enabled({}) is False
    assert conditioning_enabled(None) is False
    assert conditioning_enabled({"filter_cutoff_hz": 5000.0}) is True
    assert conditioning_enabled({"notch_enabled": True}) is True


# ================================================================================================
# 2. the sentence that names the chain
# ================================================================================================
def test_chain_description_names_the_stages_in_the_order_they_are_applied():
    """Stage order is the recipe, not a list: notch before detrend before the low-pass is
    a different trace from any other ordering, and this sentence is what a user checks the
    export against."""
    text = chain_description(CHAINS["full chain"])
    order = [text.index(s) for s in ("notch", "Savitzky-Golay", "low-pass")]
    assert order == sorted(order)
    assert "60 Hz" in text and "5000 Hz" in text and "order 8" in text
    assert chain_description({}) == ""
    assert chain_description(None) == ""


def test_an_explicit_notch_line_list_is_described_by_its_lines_not_by_a_series():
    """The two notch spellings are mutually exclusive in the kernel, so the sentence reads
    the same one the filter does (``modules.notch_freqs_param``) — a description that
    merged them would name a filter nobody ran."""
    text = chain_description({"notch_enabled": True, "notch_freqs": [60.0, 180.0],
                              "notch_freq_hz": 50.0, "notch_harmonics": 4})
    assert "60/180 Hz" in text
    assert "50" not in text


# ================================================================================================
# 3. the basis column is additive
# ================================================================================================
def test_a_frame_that_never_had_the_basis_column_gains_not_recorded():
    """An OLD project's measurement table has no ``signal_basis``. It must load, and its
    rows must read as UNKNOWN — those rows were raw, but nobody wrote that down, and a
    stamp the app invents is not a record."""
    old = pd.DataFrame({"measurement_id": [0, 1], "amplitude": [30.0, 31.0]})
    out = ensure_signal_basis_column(old.copy())
    assert list(out[SIGNAL_BASIS_COLUMN]) == [BASIS_UNKNOWN, BASIS_UNKNOWN]

    # ...and it never overwrites a stamp that IS there (idempotent, called on every append).
    already = out.copy()
    already[SIGNAL_BASIS_COLUMN] = pd.Series([BASIS_PROCESSED, BASIS_UNKNOWN],
                                             dtype="string")
    assert list(ensure_signal_basis_column(already)[SIGNAL_BASIS_COLUMN]) == \
        [BASIS_PROCESSED, BASIS_UNKNOWN]


# ================================================================================================
# 4. the export: samples, time base, and a chain that can be rerun
# ================================================================================================
def test_the_export_writes_the_samples_and_a_time_base_from_the_recordings_first_sample(
        tmp_path):
    raw = _trace()
    params = CHAINS["detrend + lowpass"]
    data = conditioned_trace(raw, FS, params)
    out = write_processed_trace(tmp_path / "proc.csv", data, FS, units="pA",
                                params=params, recording="synthetic.dat", channel="current",
                                channel_index=0, t0_s=12.5, span="whole recording")

    assert out["data"].exists() and out["meta"].exists()
    assert tuple(processed_trace_export_paths(tmp_path / "proc.csv")) == \
        (out["data"], out["meta"])

    df = pd.read_csv(out["data"])
    assert list(df.columns) == ["time_s", "current_pA"]
    assert len(df) == data.size
    # The samples round-trip EXACTLY: %.9g carries every float32 back unchanged, so an
    # exported trace is the trace, not a rounded picture of it.
    assert np.array_equal(df["current_pA"].to_numpy(dtype=np.float32), data)
    assert np.allclose(df["time_s"].to_numpy(),
                       np.arange(data.size) / FS, rtol=0, atol=1e-12)


def test_the_sidecar_carries_enough_to_rebuild_the_very_same_array(tmp_path):
    """Provenance is only provenance if it REPRODUCES: the sidecar's own ``params``, fed
    back into ``conditioned_trace``, must give the exported samples bit for bit."""
    raw = _trace()
    params = CHAINS["full chain"]
    data = conditioned_trace(raw, FS, params)
    out = write_processed_trace(tmp_path / "proc.csv", data, FS, units="pA",
                                params=params, recording="synthetic.dat", t0_s=0.0,
                                span="whole recording")

    meta = json.loads(out["meta"].read_text())
    assert meta["signal_basis"] == BASIS_PROCESSED
    assert meta["sample_rate_hz"] == FS and meta["units"] == "pA"
    assert meta["n_samples"] == data.size and meta["span"] == "whole recording"
    assert meta["chain"] == chain_description(params)
    assert meta["data_file"] == out["data"].name

    again = conditioned_trace(raw, meta["sample_rate_hz"], meta["params"])
    assert _digest(again) == _digest(data)


def test_a_slice_export_says_where_in_the_recording_it_came_from(tmp_path):
    """A cursor-selection export is a WINDOW on a recording, so its time base counts from
    the recording's first sample and its sidecar names the index range. Without that, two
    exports of two different events are indistinguishable files of numbers."""
    raw = _trace()
    params = CHAINS["lowpass only"]
    data = conditioned_trace(raw, FS, params)
    start, stop = 40_000, 44_000
    out = write_processed_trace(tmp_path / "cut.csv", data[start:stop], FS, units="pA",
                                start_idx=start, params=params, span="cursor selection")

    df = pd.read_csv(out["data"])
    assert len(df) == stop - start
    assert df["time_s"].iloc[0] == pytest.approx(start / FS)
    meta = json.loads(out["meta"].read_text())
    assert (meta["start_idx"], meta["end_idx"]) == (start, stop)
    assert meta["span"] == "cursor selection"


def test_a_failed_export_leaves_no_half_written_csv(tmp_path):
    """Staged, like every export over user data: a truncated CSV with a plausible header
    is worse than no file, because it opens."""
    target = tmp_path / "nope" / "proc.csv"          # the directory does not exist
    with pytest.raises(Exception):
        write_processed_trace(target, _trace(), FS, units="pA", params={})
    assert not target.exists()
