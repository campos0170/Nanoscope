"""USABLE-2/3: what the kernels' SHIPPED DEFAULTS do on the OWNER'S OWN recording.

The section header and the `detect_derivative` / `detect_peaks` / `detect_tkeo`
docstrings now quote recall, precision and dwell against `detect_events`' own answer
on the 15 s apo-ferritin file. Those are the numbers a user would get if they picked
one of these kernels off the menu, so they are the numbers a test has to hold: a
caption is a claim, and a claim about a recording is checked on that recording.

Two kinds of test here:

* THE MEASUREMENT, re-scored on the real file (skipped when it is not on this
  machine): every number quoted in a docstring, reproduced.
* THE SENTENCE, checked always: a default cannot move without the claim about it
  moving too.

Set ``NANOSCOPE_DETECT_MODULE`` to another copy of ``detect.py`` to run these against
it -- that is how they were shown to fail against the pre-fix file.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import nanoscope.analysis.detect as _detect

APO15 = Path(
    "/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_0004_process "
    "_ -1000 mV_ 385-400 s time trace.abf"
)
FS = 50_000.001192092925


def _module():
    path = os.environ.get("NANOSCOPE_DETECT_MODULE")
    if not path:
        return _detect
    name = "nanoscope.analysis._detect_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def D():
    return _module()


@pytest.fixture(scope="module")
def apo(D):
    """The owner's 15 s apo recording, the app's answer on it, and one basis.

    ONE `DetectionBasis` for every kernel, which is the comparison the section is
    built to make: whatever the kernels disagree about, it is not the baseline.
    """
    if not APO15.exists():
        pytest.skip("the owner's 15 s apo recording is not on this machine")
    from nanoscope.core.types import ChannelKind
    from nanoscope.io.base import open_document

    doc = open_document(str(APO15))
    cur = [c for c in doc.meta.channels if c.kind is ChannelKind.CURRENT]
    raw = np.asarray(doc.channels[cur[0].index], np.float32)
    fs = float(doc.meta.sample_rate_hz)
    df, _prov = D.detect_events(raw, fs)
    ref = (
        df["start_idx"].to_numpy().astype(np.int64),
        df["end_idx"].to_numpy().astype(np.int64),
    )
    return raw, fs, df, ref, D.detection_basis(raw, fs)


def _score(ref, starts, ends, fs):
    """Recall over reference events, precision over returned spans, by overlap."""
    ref_s, ref_e = ref
    hit_ref = np.zeros(ref_s.size, bool)
    hit_det = np.zeros(starts.size, bool)
    j = 0
    for i in range(ref_s.size):
        while j < starts.size and ends[j] <= ref_s[i]:
            j += 1
        k = j
        while k < starts.size and starts[k] < ref_e[i]:
            hit_ref[i] = True
            hit_det[k] = True
            k += 1
    dwell_ms = (ends - starts) / fs * 1e3
    return (
        float(hit_ref.mean()),
        float(hit_det.mean()) if starts.size else float("nan"),
        float(np.median(dwell_ms)) if dwell_ms.size else float("nan"),
        float(dwell_ms.max()) if dwell_ms.size else float("nan"),
    )


# ---------------------------------------------------------------------------
# the reference the whole section is scored against
# ---------------------------------------------------------------------------
def test_the_apps_own_answer_on_the_recording_is_what_the_header_says_it_is(apo):
    _raw, _fs, df, _ref, _basis = apo
    assert len(df) == 115
    assert np.median(df["dwell_s"]) * 1e3 == pytest.approx(20.1, abs=0.2)
    assert df["dwell_s"].max() * 1e3 == pytest.approx(33.9, abs=0.2)
    assert np.median(df["depth_pA"]) == pytest.approx(90.3, abs=0.5)


# ---------------------------------------------------------------------------
# USABLE-2: the derivative kernel is not user-ready at its defaults here
# ---------------------------------------------------------------------------
def test_the_derivative_defaults_return_a_413_ms_span_and_two_thirds_recall(D, apo):
    raw, fs, _df, ref, basis = apo
    starts, ends = D.detect_derivative(raw, fs, basis=basis)
    recall, precision, median_ms, longest_ms = _score(ref, starts, ends, fs)
    assert starts.size == 146
    assert recall == pytest.approx(0.678, abs=0.005)
    assert precision == pytest.approx(1.000, abs=0.005)
    assert median_ms == pytest.approx(4.49, abs=0.05)
    assert longest_ms == pytest.approx(412.9, abs=0.5)
    assert longest_ms > 12 * np.median((ref[1] - ref[0]) / fs * 1e3), (
        "the shipped defaults emit a span an order of magnitude longer than any "
        "real event on this file -- that is the sentence the docstring must carry"
    )


def test_capping_the_pair_dwell_bounds_the_span_and_costs_recall_by_the_quoted_amount(
    D, apo
):
    raw, fs, _df, ref, basis = apo
    starts, ends = D.detect_derivative(raw, fs, basis=basis, max_pair_dwell_s=50e-3)
    recall, _prec, _med, longest_ms = _score(ref, starts, ends, fs)
    assert longest_ms == pytest.approx(28.7, abs=0.5)
    assert recall == pytest.approx(0.557, abs=0.005)


def test_the_derivative_docstring_reports_that_recording_and_says_what_to_tune(D):
    doc = D.detect_derivative.__doc__
    assert "OWNER'S OWN RECORDING" in doc and "NEEDS TUNING" in doc
    assert "412.9 ms" in doc and "recall 0.678" in doc
    assert "max_pair_dwell_s=50e-3" in doc and "lowpass" in doc
    assert "0.557" in doc and "0.930" in doc and "0.362" in doc


# ---------------------------------------------------------------------------
# USABLE-3: peak widths are not threshold crossings
# ---------------------------------------------------------------------------
def test_peak_defaults_are_right_about_where_and_wrong_about_how_long(D, apo):
    raw, fs, df, ref, basis = apo
    starts, ends = D.detect_peaks(raw, fs, basis=basis)
    recall, precision, median_ms, _longest = _score(ref, starts, ends, fs)
    assert starts.size == 278
    assert starts.size / len(df) == pytest.approx(2.42, abs=0.01)
    assert recall == pytest.approx(0.852, abs=0.005)
    assert precision == pytest.approx(1.000, abs=0.005)
    assert median_ms == pytest.approx(0.44, abs=0.02)
    assert median_ms < 0.05 * np.median(df["dwell_s"]) * 1e3


def test_rel_height_is_the_knob_that_makes_the_dwell_comparable(D, apo):
    raw, fs, df, ref, basis = apo
    starts, ends = D.detect_peaks(raw, fs, basis=basis, rel_height=0.9)
    recall, precision, median_ms, _longest = _score(ref, starts, ends, fs)
    assert starts.size == 91
    assert recall == pytest.approx(1.000, abs=0.005)
    assert precision == pytest.approx(1.000, abs=0.005)
    assert median_ms == pytest.approx(24.9, abs=0.2)
    ref_median_ms = float(np.median(df["dwell_s"]) * 1e3)
    assert 0.8 * ref_median_ms < median_ms < 1.5 * ref_median_ms


def test_the_default_rel_height_is_still_the_right_one_for_a_single_sample_spike(D):
    """Why 0.9 is a recipe and not the default: it inflates the shape this is for."""
    rng = np.random.default_rng(7)
    x = 1000.0 + rng.normal(0.0, 5.0, 100_000)
    for i in range(40):
        x[5_000 + i * 2_000] -= 100.0
    x = x.astype(np.float32)
    basis = D.detection_basis(np.asarray(x, dtype=np.float64), 50_000.0)
    narrow = D.detect_peaks(x, 50_000.0, basis=basis, rel_height=0.5)
    wide = D.detect_peaks(x, 50_000.0, basis=basis, rel_height=0.9)
    assert float(np.median(narrow[1] - narrow[0])) == 2.0
    assert float(np.median(wide[1] - wide[0])) >= 11.0


def test_the_peaks_docstring_reports_that_recording_and_names_the_knob(D):
    doc = D.detect_peaks.__doc__
    assert "OWNER'S OWN RECORDING" in doc
    assert "2.42 spans per reference event" in doc
    assert "0.44 ms" in doc and "20.1 ms" in doc
    assert "rel_height" in doc and "24.9 ms" in doc and "11-sample" in doc


# ---------------------------------------------------------------------------
# TKEO on the same recording, and the menu that sends a user to one of them
# ---------------------------------------------------------------------------
def test_tkeo_needs_link_gap_on_events_this_long_and_the_docstring_says_so(D, apo):
    raw, fs, _df, ref, basis = apo
    bare = D.detect_tkeo(raw, fs, basis=basis)
    assert bare[0].size == 14_006
    linked = D.detect_tkeo(raw, fs, basis=basis, link_gap_s=2e-3)
    recall, precision, median_ms, _longest = _score(ref, *linked, fs)
    assert linked[0].size == 108
    assert recall == pytest.approx(0.826, abs=0.005)
    assert precision == pytest.approx(0.796, abs=0.005)
    assert median_ms == pytest.approx(19.6, abs=0.2)
    doc = D.detect_tkeo.__doc__
    assert "OWNER'S OWN RECORDING" in doc and "14,006 spans" in doc
    assert "link_gap_s=2e-3" in doc


def test_the_polarity_gate_removes_nothing_on_this_recording_as_claimed(D, apo):
    raw, fs, _df, _ref, basis = apo
    on = D.detect_tkeo(raw, fs, basis=basis, polarity_gate=True)
    off = D.detect_tkeo(raw, fs, basis=basis, polarity_gate=False)
    assert np.array_equal(on[0], off[0]) and np.array_equal(on[1], off[1])
    assert "removes nothing at all on this file" in D.detect_tkeo.__doc__


def test_the_fixed_threshold_pays_in_precision_on_this_drifting_pore(D, apo):
    raw, fs, _df, ref, basis = apo
    b = float(np.median(basis.baseline))
    starts, ends = D.detect_fixed_threshold(
        raw,
        fs,
        level=b + basis.polarity * 5.0 * basis.noise,
        exit_level=b + basis.polarity * 2.0 * basis.noise,
    )
    recall, precision, median_ms, _longest = _score(ref, starts, ends, fs)
    assert starts.size == 158
    assert recall == pytest.approx(0.896, abs=0.005)
    assert precision == pytest.approx(0.627, abs=0.005)
    assert median_ms == pytest.approx(15.5, abs=0.2)
    assert "PRECISION 0.627" in D.detect_fixed_threshold.__doc__


def test_the_section_header_names_the_two_kernels_that_need_tuning(D):
    src = inspect.getsource(D)
    header = src[src.index("conventional detector kernels") : src.index("NOISE_METHODS:")]
    assert "NEED TUNING" in header
    assert "detect_derivative" in header and "detect_peaks" in header
    assert "115 events" in header and "412.9 ms" in header
    assert "detect_events` remains the default detector" in header
