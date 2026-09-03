"""The kernel section's CLAIMS, checked against the kernels' BEHAVIOUR.

Every test here takes a sentence that `analysis.detect`'s conventional-kernel section
asserts and runs the code it is about. A caption is a claim: if one of these fails,
either the sentence or the kernel is wrong, and the test says which.

The module under test is normally `nanoscope.analysis.detect`. Set
``NANOSCOPE_DETECT_MODULE`` to the path of another copy of that file -- the pre-fix
copy, say -- and every test here runs against THAT instead, which is how these were
shown to fail before the fixes rather than after them.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys

import numpy as np
import pytest

import nanoscope.analysis.detect as _detect

FS = 50_000.0
I0 = 1000.0
DEPTH = 100.0


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
def noiseless_pulse():
    """A LITERALLY noiseless square pulse: constant 1000 pA, [500, 600) at 900 pA."""
    x = np.full(2000, I0)
    x[500:600] -= DEPTH
    return x.astype(np.float32)


@pytest.fixture(scope="module")
def quiet_pulse():
    """The same pulse in 0.01 pA of noise -- the trace the docstrings now cite."""
    rng = np.random.default_rng(0)
    x = np.full(2000, I0) + rng.normal(0.0, 0.01, 2000)
    x[500:600] -= DEPTH
    return x.astype(np.float32)


def _both_sign_spikes(seed=0, n=40_000):
    """Blockade spikes plus wrong-sign artifacts -- the trace the polarity gate is for."""
    rng = np.random.default_rng(seed)
    x = I0 + rng.normal(0.0, 5.0, n)
    for i in range(40):
        x[1000 + i * 900 : 1003 + i * 900] -= DEPTH
    for i in range(20):
        x[500 + i * 1800 : 503 + i * 1800] += DEPTH
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# PC-NOISELESS: a worked example in a docstring has to be runnable
# ---------------------------------------------------------------------------
def test_no_kernel_promises_a_worked_example_on_a_trace_it_would_refuse(D):
    """Only the kernel that needs no noise estimate may cite a NOISELESS example."""
    for fn in (D.detect_peaks, D.detect_derivative, D.detect_tkeo):
        doc = (fn.__doc__ or "").lower()
        for phrase in ("noiseless square pulse", "noiseless pulse"):
            assert phrase not in doc, (
                f"{fn.__name__} cites a {phrase!r} example, but it goes through "
                "detection_basis, which refuses a trace whose MAD is 0 -- the "
                "example raises instead of detecting"
            )


def test_the_baseline_relative_kernels_really_do_refuse_a_noiseless_trace(
    D, noiseless_pulse
):
    """Why the sentence above matters: the example in the old docstrings raised."""
    for fn in (D.detect_peaks, D.detect_derivative, D.detect_tkeo):
        with pytest.raises(ValueError, match="no measurable noise"):
            fn(noiseless_pulse, FS)


def test_the_fixed_threshold_keeps_its_noiseless_example_because_it_runs(
    D, noiseless_pulse
):
    starts, ends = D.detect_fixed_threshold(
        noiseless_pulse, FS, level=I0 - DEPTH / 2, exit_level=I0 - DEPTH / 10
    )
    assert (starts.tolist(), ends.tolist()) == ([500], [600])
    assert "noiseless" in D.detect_fixed_threshold.__doc__.lower()


def test_the_quiet_pulse_examples_the_docstrings_now_cite_return_what_they_claim(
    D, quiet_pulse
):
    s, e = D.detect_derivative(quiet_pulse, FS)
    assert (s.tolist(), e.tolist()) == ([500], [600])
    s, e = D.detect_peaks(quiet_pulse, FS)
    assert (s.tolist(), e.tolist()) == ([499], [600])
    for fn in (D.detect_peaks, D.detect_derivative):
        assert "0.01 pA" in fn.__doc__


# ---------------------------------------------------------------------------
# PC-C4: the section header's "they all measure the same way"
# ---------------------------------------------------------------------------
def test_the_fixed_threshold_needs_no_baseline_door_and_the_other_three_do(D, monkeypatch):
    """The header's claim, executed: disable the door and see who still runs."""
    def refuse(*_a, **_k):
        raise AssertionError("baseline door called")

    monkeypatch.setattr(D, "characterisation_baseline", refuse)
    x = _both_sign_spikes()
    starts, _ends = D.detect_fixed_threshold(x, FS, level=I0 - DEPTH / 2)
    assert starts.size > 0, "the absolute-level kernel must not need the baseline door"
    for fn in (D.detect_peaks, D.detect_derivative, D.detect_tkeo):
        with pytest.raises(AssertionError, match="baseline door"):
            fn(x, FS)


def test_the_section_header_does_not_claim_all_five_share_the_doors(D):
    """`detect_fixed_threshold` takes no baseline, no polarity and no noise."""
    src = inspect.getsource(D)
    header = src[src.index("conventional detector kernels"):src.index("NOISE_METHODS:")]
    assert "ALL FIVE MEASURE THE SAME WAY" not in header
    assert "Every kernel below takes its baseline" not in header
    assert "detect_fixed_threshold" in header


def test_the_header_does_not_claim_detect_events_returns_starts_and_ends(D):
    """`detect_events` returns (DataFrame, provenance); the four kernels return spans."""
    src = inspect.getsource(D)
    header = src[src.index("conventional detector kernels"):src.index("NOISE_METHODS:")]
    assert "ALL FIVE RETURN THE SAME THING" not in header
    x = _both_sign_spikes()
    out = D.detect_events(np.asarray(x, dtype=np.float64), FS)
    assert len(out) == 2 and hasattr(out[0], "columns")
    for fn in (D.detect_peaks, D.detect_derivative, D.detect_tkeo):
        s, e = fn(x, FS)
        assert s.dtype == np.int64 and e.dtype == np.int64


# ---------------------------------------------------------------------------
# PC-REDUCEAT: the comment promised a vectorised gate
# ---------------------------------------------------------------------------
def test_the_polarity_gate_is_the_vectorised_pass_its_comment_describes(D):
    body = inspect.getsource(D.detect_tkeo).split('"""')[-1]
    assert "np.add.reduceat" in body, "the gate's comment promises a reduceat"
    assert "for s, e in zip(starts, ends)" not in body, (
        "the gate is still a Python loop per event"
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("link_gap_s", [None, 1e-4])
def test_the_vectorised_gate_keeps_exactly_the_runs_the_loop_kept(D, seed, link_gap_s):
    """The reference implementation is the loop it replaced, written out here."""
    x = _both_sign_spikes(seed=seed)
    basis = D.detection_basis(np.asarray(x, dtype=np.float64), FS)
    ungated_s, ungated_e = D.detect_tkeo(
        x, FS, basis=basis, link_gap_s=link_gap_s, polarity_gate=False
    )
    expected = [
        (int(s), int(e))
        for s, e in zip(ungated_s, ungated_e)
        if basis.dev[s:e].mean() > 0.0
    ]
    s, e = D.detect_tkeo(x, FS, basis=basis, link_gap_s=link_gap_s, polarity_gate=True)
    assert list(zip(s.tolist(), e.tolist())) == expected
    assert s.size < ungated_s.size, "the gate must still remove wrong-sign bursts"


def test_the_gate_survives_a_run_that_touches_the_last_sample(D):
    """reduceat over the run bounds needs a padded array when ``end == n``."""
    x = np.full(600, I0, dtype=np.float64)
    rng = np.random.default_rng(5)
    x += rng.normal(0.0, 1.0, x.size)
    x[-6:] -= 200.0                                  # an event running off the end
    s, e = D.detect_tkeo(x.astype(np.float32), FS, k=20.0)
    assert s.size >= 1
    assert int(e[-1]) <= x.size
