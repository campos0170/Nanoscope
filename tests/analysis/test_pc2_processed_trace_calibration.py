"""PC-REPRO: the processed-trace sidecar is a RECIPE, or it should not say it is one.

``processed_trace_provenance`` promised "enough to REPRODUCE the array, not merely to
admire it" and left out the one fact that is not in ``params``: the calibration. The chain
runs on the CALIBRATED channel (``session.calibrated(doc, ch)``), so the same recording,
the same params and a different gain are two different arrays — and before this change
they wrote two provenance blocks identical in every field but the file name. The
2026-09-02 probe measured it: a reader handed the stored channel and the sidecar could
only assume gain 1, and missed by 923 pA.

These tests hold the promise rather than soften it:

  - the sidecar plus the STORED channel rebuild the exported samples bit for bit, under a
    non-identity calibration, whole recording and cursor slice alike;
  - two calibrations of one recording no longer produce interchangeable sidecars;
  - an unrecorded calibration is ``null`` — "nobody wrote this down" — and is never
    readable as gain 1, the same discipline as ``BASIS_UNKNOWN``;
  - no NaN ever reaches the JSON, because a sidecar a strict reader rejects is a sidecar.

They FAIL against the pre-fix module: it emitted no ``cal_gain``/``cal_offset`` at all.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.conditioning import (
    conditioned_trace,
    processed_trace_provenance,
    write_processed_trace,
)
from nanoscope.core.calibration import ChannelCalibration

FS = 50_000.0
N = 60_000

#: A chain with memory in every stage (block medians, filtfilt), so a slice conditioned on
#: its own is measurably NOT the slice of the whole-recording chain — which is what makes
#: the `[start_idx:end_idx]` half of the recipe load-bearing rather than decorative.
CHAIN = {
    "notch_enabled": True, "notch_freq_hz": 60.0, "notch_q": 30.0, "notch_harmonics": 2,
    "detrend_enabled": True, "detrend_method": "block_median", "detrend_window_s": 0.05,
    "filter_cutoff_hz": 5000.0, "filter_order": 8, "filter_kind": "bessel",
}

#: Not gain 1 and not offset 0: an identity calibration would pass every one of these
#: tests against the pre-fix module, which is exactly how the defect survived.
CAL = ChannelCalibration(gain=10.0, offset=-3.5, target_units="pA")


def _stored() -> np.ndarray:
    """The channel as it sits in the file: float32, uncalibrated."""
    rng = np.random.default_rng(4)
    x = (100.0 + 1.5 * rng.normal(size=N)).astype(np.float32)
    x[20_000:20_400] -= np.float32(40.0)          # an event, so depth-scale errors show
    return x


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def _rebuild(stored: np.ndarray, meta: dict) -> np.ndarray:
    """THE RECIPE, written out of the sidecar alone — no access to the app's state.

    This is the whole point of the export: a reader with the recording and the JSON must
    be able to get back to the numbers. Note the order — calibrate, condition the WHOLE
    channel, then slice — because every stage is a filter with memory.
    """
    assert meta["cal_gain"] is not None and meta["cal_offset"] is not None, \
        "the sidecar does not name the calibration; the array cannot be rebuilt"
    raw = stored * np.float32(meta["cal_gain"]) + np.float32(meta["cal_offset"])
    whole = conditioned_trace(raw, meta["sample_rate_hz"], meta["params"])
    return whole[int(meta["start_idx"]):int(meta["end_idx"])]


def _export(tmp_path, name, stored, cal, *, start=0, stop=None, **kw):
    stop = N if stop is None else stop
    data = conditioned_trace(cal.apply(stored), FS, CHAIN)[start:stop]
    gain, offset = cal.fingerprint()
    out = write_processed_trace(
        tmp_path / f"{name}.csv", data, FS, units="pA", start_idx=start, params=CHAIN,
        recording="A.dat", channel="current", channel_index=0, t0_s=0.0,
        span="whole recording" if start == 0 and stop == N else "cursor selection",
        **{"cal_gain": gain, "cal_offset": offset, **kw})
    return data, out, json.loads(out["meta"].read_text())


# ================================================================================================
# 1. the promise
# ================================================================================================
def test_the_sidecar_and_the_stored_channel_rebuild_the_export_bit_for_bit(tmp_path):
    """"Enough to reproduce" means a reader can do it, from the file and the JSON."""
    stored = _stored()
    data, out, meta = _export(tmp_path, "whole", stored, CAL)

    assert meta["cal_gain"] == 10.0 and meta["cal_offset"] == -3.5
    assert _digest(_rebuild(stored, meta)) == _digest(data)

    # ...and the CSV a reader actually opens holds those same samples.
    csv = pd.read_csv(out["data"])["current_pA"].to_numpy(dtype=np.float32)
    assert _digest(csv) == _digest(data)


def test_a_cursor_slice_rebuilds_too_and_only_via_the_whole_recording_chain(tmp_path):
    """``start_idx``/``end_idx`` are part of the recipe, not just a label.

    Conditioning the cut on its own is a different array at its edges — the second
    assertion is what stops the recipe being read as "condition what you exported"."""
    stored = _stored()
    start, stop = 19_000, 23_000
    data, _out, meta = _export(tmp_path, "cut", stored, CAL, start=start, stop=stop)

    assert (meta["start_idx"], meta["end_idx"]) == (start, stop)
    assert _digest(_rebuild(stored, meta)) == _digest(data)

    on_the_cut = conditioned_trace(CAL.apply(stored)[start:stop], FS, CHAIN)
    assert _digest(on_the_cut) != _digest(data)


def test_two_calibrations_of_one_recording_no_longer_write_the_same_provenance(tmp_path):
    """The skeptic's scenario, as a test.

    Gain 1 and gain 10 over the same file and the same params: two arrays a factor of ten
    apart, whose sidecars used to differ only in ``data_file``. Each must now rebuild
    ITSELF and fail to rebuild the other."""
    stored = _stored()
    a_data, _a, a_meta = _export(tmp_path, "cal_default", stored, ChannelCalibration())
    b_data, _b, b_meta = _export(tmp_path, "cal_gain10", stored, CAL)

    assert _digest(a_data) != _digest(b_data)
    differing = {k for k in set(a_meta) | set(b_meta) if a_meta.get(k) != b_meta.get(k)}
    assert {"cal_gain", "cal_offset"} <= differing, \
        f"the calibration is not in the sidecar; only {differing} tell the two apart"

    assert _digest(_rebuild(stored, a_meta)) == _digest(a_data)
    assert _digest(_rebuild(stored, b_meta)) == _digest(b_data)
    assert _digest(_rebuild(stored, a_meta)) != _digest(b_data)


# ================================================================================================
# 2. the honesty of an absent stamp
# ================================================================================================
def test_an_unrecorded_calibration_is_null_and_is_never_readable_as_gain_one(tmp_path):
    """A caller that did not stamp the calibration must not look like one that stamped 1.

    Same rule as ``BASIS_UNKNOWN``: "nobody wrote this down" and "this was the identity"
    are different facts, and inventing the second is how a record stops being a record."""
    data = conditioned_trace(CAL.apply(_stored()), FS, CHAIN)
    out = write_processed_trace(tmp_path / "unstamped.csv", data, FS, units="pA",
                                params=CHAIN, span="whole recording")
    meta = json.loads(out["meta"].read_text())

    assert "cal_gain" in meta and "cal_offset" in meta      # the keys are always present
    assert meta["cal_gain"] is None and meta["cal_offset"] is None
    with pytest.raises(AssertionError):
        _rebuild(_stored(), meta)                           # and it says so, loudly


def test_a_nonfinite_stamp_is_recorded_as_unknown_rather_than_as_nan_in_the_json(tmp_path):
    """``json.dump`` writes a bare ``NaN`` literal, which is not JSON.

    A sidecar a strict reader rejects has failed at the one job it has, so a NaN gain,
    offset or ``t0_s`` becomes ``null`` — the same "not recorded" the absent stamp gets."""
    prov = processed_trace_provenance(
        params=CHAIN, fs=FS, n_samples=10, start_idx=0, units="pA",
        cal_gain=float("nan"), cal_offset=float("inf"), t0_s=float("nan"))
    assert prov["cal_gain"] is None and prov["cal_offset"] is None and prov["t0_s"] is None

    text = json.dumps(prov)
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text, parse_constant=_reject)                # strict: raises on NaN/Infinity


def _reject(token: str):  # pragma: no cover - only called if a bare literal got through
    raise ValueError(f"non-JSON literal in the sidecar: {token}")


def test_the_stamp_is_the_calibration_fingerprint_and_nothing_more(tmp_path):
    """Gain and offset, because they are the only two numbers that alter the ARRAY.

    ``target_units``/``kind``/``note`` are labels — ``ChannelCalibration.fingerprint``
    draws that line already, and the sidecar draws it in the same place rather than
    inventing a second answer to the same question."""
    labelled = ChannelCalibration(gain=10.0, offset=-3.5, target_units="nA",
                                  note="hand-entered")
    assert labelled.fingerprint() == CAL.fingerprint()

    stored = _stored()
    _a_data, _a, a_meta = _export(tmp_path, "plain", stored, CAL)
    _b_data, _b, b_meta = _export(tmp_path, "labelled", stored, labelled)
    assert (a_meta["cal_gain"], a_meta["cal_offset"]) == \
           (b_meta["cal_gain"], b_meta["cal_offset"])
