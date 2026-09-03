"""C4 + C8 + C41 — a non-finite value must never come back as a confident number.

The defect, end to end: a CSV with ONE blank cell loads as one NaN sample. Welch averages
segment periodograms, so that single sample makes EVERY bin of the spectrum NaN; ``band_rms``
filtered the non-finite bins out, found fewer than two left, and returned ``0.0``. The panel
printed **"Total RMS noise: 0 pA"** for a trace whose true σ was 1.49 pA — a perfect-looking
QC result — and the load reported ``issues: ()``.

Three invariants, each in the one place every path must cross:

* :meth:`TraceDocument.__post_init__` — every reader builds one, so every reader reports.
* :class:`ChannelCalibration.__post_init__` — the sidecar, the project and the Gain box all
  build one, so a non-finite gain is refused once instead of four times (and ``1e999`` typed
  into the panel is the same defect as ``NaN`` in a sidecar).
* :func:`band_rms` — every RMS read-out goes through it, so "cannot be measured" stops being
  spelled ``0.0``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from nanoscope.analysis.psd import band_rms, noise_summary, welch_psd
from nanoscope.core.calibration import (
    CalibrationSet,
    ChannelCalibration,
    load_sidecar,
    sidecar_path,
)
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
    scan_nonfinite,
)
from nanoscope.io.base import open_document

FS = 100_000.0
SIGMA = 1.4933          # the σ from the original repro
ABF = Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0006_Process_-600 mV.abf")
_needs_abf = pytest.mark.skipif(not ABF.exists(), reason="owner's -600 mV recording not present")


# ================================================================================================
# builders
# ================================================================================================
def _noise(n=200_000, seed=0):
    return np.random.default_rng(seed).normal(0.0, SIGMA, n)


def _doc(tmp_path, arrays, *, names=None, name="rec.dat") -> TraceDocument:
    names = names or [f"ch{i}" for i in range(len(arrays))]
    infos = tuple(ChannelInfo(index=i, name=nm, raw_units="pA", kind=ChannelKind.CURRENT)
                  for i, nm in enumerate(names))
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic",
                         n_samples=int(arrays[0].size), sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=infos)
    return TraceDocument(meta=meta,
                         channels=tuple(np.ascontiguousarray(a, np.float32) for a in arrays))


def _csv(tmp_path, *, blank_row: int | None, n=20_000, name="trace.csv") -> Path:
    t = np.arange(n) / FS
    y = _noise(n, seed=1)
    lines = ["time,current"] + [f"{ti:.9g},{yi:.9g}" for ti, yi in zip(t, y)]
    if blank_row is not None:
        lines[blank_row + 1] = f"{t[blank_row]:.9g},"      # ONE blank cell
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


# ================================================================================================
# C41 — band_rms
# ================================================================================================
def test_c41_a_poisoned_spectrum_reads_as_NaN_not_as_zero():
    """THE regression. One NaN sample; pre-fix this returned 0.0 with no warning."""
    x = _noise()
    x[12_345] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f, p = welch_psd(x, FS)
    assert not np.isfinite(p).any(), "precondition: Welch poisons every bin"

    with pytest.warns(RuntimeWarning, match="non-finite"):
        rms = band_rms(f, p, 0.0, FS / 2)
    assert np.isnan(rms), f"a poisoned band reported {rms!r} — 0.0 reads as a perfect pore"


def test_c41_the_clean_signal_still_reports_its_true_rms():
    """The fix must not cost the correct answer."""
    x = _noise()
    f, p = welch_psd(x, FS)
    assert band_rms(f, p, 0.0, FS / 2) == pytest.approx(float(np.std(x)), rel=0.02)


def test_c41_a_partially_poisoned_band_is_also_NaN():
    """Skipping the bad bins and integrating the rest understates the noise by exactly the
    part that is broken — a quieter version of the same lie."""
    f = np.linspace(0.0, 50_000.0, 200)
    p = np.ones_like(f)
    p[50] = np.nan
    with pytest.warns(RuntimeWarning, match=r"1 of \d+"):
        assert np.isnan(band_rms(f, p, 0.0, 50_000.0))


def test_c41_a_genuinely_empty_band_is_still_an_honest_zero():
    """The distinction the whole fix rests on: EMPTY is 0.0, UNMEASURABLE is NaN."""
    f = np.linspace(0.0, 50_000.0, 100)
    p = np.ones_like(f)
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # these must not warn
        assert band_rms(f, p, 0.0, 0.0) == 0.0             # degenerate
        assert band_rms(f, p, 50_000.0, 80_000.0) == 0.0   # wholly above Nyquist
        assert band_rms(f, p, 8_000.0, 5_000.0) == 0.0     # inverted
        assert band_rms(np.array([]), np.array([]), 0.0, 1.0) == 0.0


def test_c41_a_negative_integral_is_not_a_power_spectrum():
    """sqrt of a negative variance used to be silently reported as 0.0."""
    f = np.linspace(0.0, 1_000.0, 50)
    p = np.full_like(f, -1.0)
    with pytest.warns(RuntimeWarning, match="negative"):
        assert np.isnan(band_rms(f, p, 0.0, 1_000.0))


def test_c41_noise_summary_carries_the_NaN_through_every_field():
    """Pre-fix: total_rms 0.0 and all three bands 0.0 — four confident wrong numbers."""
    x = _noise()
    x[7] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f, p = welch_psd(x, FS)
        s = noise_summary(f, p, FS)
    assert np.isnan(s["total_rms"])
    assert all(np.isnan(v) for v in s["bands"].values())


def test_welch_psd_says_so_at_the_source():
    """By the time a caller holds the PSD there is nothing left to point at."""
    x = _noise(50_000)
    x[100] = np.nan
    with pytest.warns(RuntimeWarning, match="non-finite sample"):
        welch_psd(x, FS)


def test_welch_psd_does_not_quietly_repair_the_signal():
    """Interpolating over the bad samples would produce a plausible spectrum for data that
    was never recorded. Reporting beats repairing."""
    x = _noise(50_000)
    x[100] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p = welch_psd(x, FS)
    assert not np.isfinite(p).any()


# ================================================================================================
# C4 — the load reports it
# ================================================================================================
def test_scan_nonfinite_counts_and_locates():
    x = np.arange(10_000, dtype=np.float32)
    assert scan_nonfinite(x) == (0, -1)
    x[4_242] = np.nan
    x[9_999] = np.inf
    assert scan_nonfinite(x) == (2, 4_242)


def test_scan_nonfinite_crosses_its_own_chunk_boundary():
    """The chunking is an implementation detail and must not be observable."""
    from nanoscope.core.types import NONFINITE_SCAN_CHUNK

    x = np.zeros(NONFINITE_SCAN_CHUNK * 2 + 7, dtype=np.float32)
    x[NONFINITE_SCAN_CHUNK + 3] = -np.inf
    assert scan_nonfinite(x) == (1, NONFINITE_SCAN_CHUNK + 3)


def test_c4_a_document_with_a_nan_sample_says_so_in_issues(tmp_path):
    """THE regression, at the invariant every reader crosses."""
    x = _noise(20_000).astype(np.float32)
    x[5_000] = np.nan
    doc = _doc(tmp_path, [x], names=["current"])

    assert doc.meta.issues, "a recording carrying NaN loaded with issues: ()"
    text = " ".join(doc.meta.issues)
    assert "non-finite" in text and "channel 0" in text and "5,000" in text


def test_c4_a_clean_document_gains_no_issue(tmp_path):
    assert _doc(tmp_path, [_noise(5_000).astype(np.float32)]).meta.issues == ()


def test_c4_the_issue_names_the_right_channel(tmp_path):
    clean = _noise(5_000).astype(np.float32)
    dirty = clean.copy()
    dirty[11] = np.inf
    doc = _doc(tmp_path, [clean, dirty], names=["IN15", "IN14"])
    assert len(doc.meta.issues) == 1
    assert "channel 1 (IN14)" in doc.meta.issues[0]


def test_c4_the_readers_own_issues_are_kept(tmp_path):
    """The invariant APPENDS; it must not swallow what the reader already reported."""
    info = ChannelInfo(index=0, name="c", raw_units="pA", kind=ChannelKind.CURRENT)
    x = np.zeros(100, dtype=np.float32)
    x[0] = np.nan
    meta = RecordingMeta(path=tmp_path / "r.dat", format_name="synthetic", n_samples=100,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,), issues=("the reader already said this",))
    doc = TraceDocument(meta=meta, channels=(x,))
    assert doc.meta.issues[0] == "the reader already said this"
    assert len(doc.meta.issues) == 2


def test_c4_rebuilding_the_document_does_not_repeat_the_issue(tmp_path):
    """A frozen value object gets rebuilt — ``Session.add_document`` does exactly this to
    fold in the sidecar's issues — and the rebuild re-enters ``__post_init__`` over a meta
    that already carries the sentence. Appending unconditionally reported the same NaN
    twice, then three times, then four."""
    from dataclasses import replace

    x = _noise(5_000).astype(np.float32)
    x[3] = np.nan
    doc = _doc(tmp_path, [x])
    assert len(doc.meta.issues) == 1

    again = replace(doc, meta=replace(doc.meta, issues=doc.meta.issues + ("a sidecar issue",)))
    assert len(again.meta.issues) == 2
    assert sum("non-finite" in i for i in again.meta.issues) == 1


def test_c4_the_session_reports_it_once_alongside_a_sidecar_issue(tmp_path):
    """The real rebuild path, end to end.

    The sidecar half asserts that a rejected entry still reaches ``meta.issues`` and still
    names the channel. Its WORDING moved with C-6: an entry naming a channel this recording
    does not have is now reported as *"the calibration saved for channel 9 was NOT
    applied — this recording has no channel 9"* rather than as *"channel '9' entry is
    malformed"*, because the entry is not malformed — the recording is a different one. The
    guarantee under test (the issue survives alongside the non-finite one, and says which
    channel) is unchanged.
    """
    from nanoscope.core.session import Session

    (tmp_path / "rec.dat").write_bytes(b"x")
    sidecar_path(tmp_path / "rec.dat").write_text('{"version": 1, "channels": {"9": '
                                                  '{"gain": 1.0, "offset": 0.0}}}')
    x = _noise(5_000).astype(np.float32)
    x[3] = np.nan
    s = Session()
    doc_id = s.add_document(_doc(tmp_path, [x], name="rec.dat"))
    issues = s.document(doc_id).meta.issues

    assert sum("non-finite" in i for i in issues) == 1
    assert any("channel 9" in i and "NOT applied" in i     # the sidecar issue survived too
               for i in issues), issues


def test_c4_a_real_csv_with_one_blank_cell_is_reported(tmp_path):
    """End to end through the actual text reader — the original repro."""
    doc = open_document(_csv(tmp_path, blank_row=5_000))
    arr = doc.raw(0)
    assert int((~np.isfinite(arr)).sum()) == 1, "precondition: pandas turns the blank into NaN"
    assert doc.meta.issues, "the CSV loaded with issues: () — C4"
    assert "non-finite" in doc.meta.issues[0]


def test_c4_a_clean_csv_is_not_flagged(tmp_path):
    doc = open_document(_csv(tmp_path, blank_row=None))
    assert doc.meta.issues == ()


def test_c4_a_segment_cut_over_the_bad_sample_inherits_the_report(tmp_path):
    """A SegmentDocument IS a TraceDocument, so the guarantee arrives without segment-specific
    code — which is the entire reason the check lives on the base class."""
    from nanoscope.core.segment import build_segment_document

    x = _noise(20_000).astype(np.float32)
    x[9_000] = np.nan
    parent = _doc(tmp_path, [x], names=["current"])
    inside = build_segment_document(parent, 8_000, 10_000, name="bad", segment_uid="s1")
    outside = build_segment_document(parent, 0, 4_000, name="good", segment_uid="s2")

    assert inside.meta.issues and "non-finite" in inside.meta.issues[0]
    assert outside.meta.issues == ()


@_needs_abf
def test_c4_the_owners_real_recording_is_clean(tmp_path):
    """The guarantee must not cry wolf on real data."""
    doc = open_document(ABF)
    assert doc.meta.issues == ()
    for i in range(len(doc.channels)):
        assert scan_nonfinite(doc.raw(i)) == (0, -1)


@_needs_abf
def test_c4_one_poisoned_sample_of_the_owners_recording_is_caught(tmp_path):
    """2,993,798 samples; one of them ruined. Both the load report and the RMS must notice."""
    doc = open_document(ABF)
    x = np.array(doc.raw(0), dtype=np.float32)
    clean_rms = float(np.std(x))
    x[1_500_000] = np.nan

    poisoned = _doc(tmp_path, [x], names=["IN15"])
    assert poisoned.meta.issues and \
        "1 of 2,993,798 samples is non-finite" in poisoned.meta.issues[0]
    # ... and it is named as a TAINT, not as the dead channel a bad gain produces.
    assert "localised taint" in poisoned.meta.issues[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f, p = welch_psd(x, 250_000.0)
        rms = band_rms(f, p, 0.0, 125_000.0)
    assert np.isnan(rms), f"reported {rms!r} for a recording whose true σ is {clean_rms:.3f} pA"


# ================================================================================================
# C8 — calibration values
# ================================================================================================
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_c8_a_non_finite_gain_is_refused_at_construction(bad):
    """One check, on the value object every loader must build."""
    with pytest.raises(ValueError, match="finite"):
        ChannelCalibration(gain=bad, offset=0.0)
    with pytest.raises(ValueError, match="finite"):
        ChannelCalibration(gain=1.0, offset=bad)


def test_c8_a_finite_calibration_is_unaffected():
    cal = ChannelCalibration(gain=0.1, offset=-3.5, target_units="mV")
    assert (cal.gain, cal.offset) == (0.1, -3.5)
    assert cal.fingerprint() == (0.1, -3.5)


def test_c8_a_non_finite_gain_would_have_destroyed_the_whole_channel():
    """Why the refusal is at construction and not at the point of use."""
    raw = np.ones(1_000, dtype=np.float32)
    unchecked = ChannelCalibration(gain=1.0)
    object.__setattr__(unchecked, "gain", float("nan"))     # what used to be constructible
    assert not np.isfinite(unchecked.apply(raw)).any()


def test_c8_the_sidecar_rejects_bare_NaN_json(tmp_path):
    """``json.loads`` accepts ``NaN``/``Infinity`` — they are not JSON, and they used to
    install straight into the CalibrationSet."""
    data = tmp_path / "rec.abf"
    data.write_bytes(b"ABF ")
    sidecar_path(data).write_text('{"version": 1, "channels": {"0": {"gain": NaN, '
                                  '"offset": 0.0}}}')
    calib = CalibrationSet()
    issues = load_sidecar(data, "doc", calib, _infos(1))
    assert issues and "NaN" in issues[0]
    assert calib.entries == {}


def _infos(n):
    return tuple(ChannelInfo(index=i, name=f"ch{i}", raw_units="pA",
                             kind=ChannelKind.CURRENT) for i in range(n))


def test_c8_the_sidecar_rejects_channels_the_recording_does_not_have(tmp_path):
    """``"-1"`` and ``"99"`` used to load with ``issues=[]`` — invisible entries under an
    index no channel has, and (for the negative one) a live collision risk."""
    data = tmp_path / "rec.abf"
    data.write_bytes(b"ABF ")
    sidecar_path(data).write_text(json.dumps({"version": 1, "channels": {
        "0": {"gain": 0.5, "offset": 0.0, "note": "real"},
        "-1": {"gain": 2.0, "offset": 0.0},
        "99": {"gain": 3.0, "offset": 0.0},
    }}))
    calib = CalibrationSet()
    issues = load_sidecar(data, "doc", calib, _infos(2))

    assert len(issues) == 2
    assert any("-1" in i for i in issues) and any("99" in i for i in issues)
    assert set(calib.entries) == {("doc", 0)}
    assert calib.entries[("doc", 0)].gain == 0.5      # the valid entry still lands


def test_c8_a_sidecar_gain_that_overflows_to_inf_is_rejected(tmp_path):
    """``1e999`` is valid JSON and parses to ``inf``; the ``parse_constant`` hook never sees
    it, so the refusal has to come from ChannelCalibration."""
    data = tmp_path / "rec.abf"
    data.write_bytes(b"ABF ")
    sidecar_path(data).write_text('{"version": 1, "channels": {"1": {"gain": 1e999, '
                                  '"offset": 0.0}}}')
    calib = CalibrationSet()
    issues = load_sidecar(data, "doc", calib, _infos(2))
    assert issues and "finite" in issues[0]
    assert calib.entries == {}


def test_c8_a_good_sidecar_still_loads_untouched(tmp_path):
    data = tmp_path / "rec.abf"
    data.write_bytes(b"ABF ")
    sidecar_path(data).write_text(json.dumps({"version": 1, "channels": {
        "0": {"gain": 0.1, "offset": 2.0, "target_units": "mV", "note": "keep"}}}))
    calib = CalibrationSet()
    assert load_sidecar(data, "doc", calib, _infos(1)) == []
    cal = calib.get("doc", 0)
    assert (cal.gain, cal.offset, cal.target_units, cal.note) == (0.1, 2.0, "mV", "keep")
