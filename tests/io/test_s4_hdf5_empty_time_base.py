"""BD4-05 — one condition, one sentence, whatever the channel count.

An acquisition armed and stopped without recording writes a structurally perfect HDF5
holding nothing. ``/current(0) + /voltage(0)`` and ``/time(0) + /current(0) +
/voltage(0)`` both got the one sentence written for that ("the file decoded but holds no
samples…"). ``/time(0) + /current(0)`` — which is exactly the layout NanoScope's own
``packaging/nanoscope_selftest.py`` writes for ``real_slice.h5`` — got instead

    /current present but no /time base and no second signal channel — this looks like a
    data export, not a recording (top-level members: current, time).

denying the /time base in the same breath as listing ``time`` among the members, and
sending the user hunting for an export tool that never made the file. Adding one more
EMPTY dataset flipped the same file to the right sentence, so one condition had two
stories decided by channel count.

Cause, and the invariant restored: the acceptance gate asked "is there a time base?" with
the function that answers "can a sample interval be measured from it?"
(``_time_signal``, which needs 2 samples). Those are two questions.
``Hdf5Reader._time_base`` answers the structural one; measurability is unchanged and
still reported as an issue with the rate left UNKNOWN, never fabricated.
"""

from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, per tests/conftest.py
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import h5py
import numpy as np
import pytest

from nanoscope.core.types import Provenance
from nanoscope.io.base import open_document, probe_path
from nanoscope.io.errors import ReaderError, UnsupportedFormatError

EMPTY = ("the file decoded but holds no samples, so there is nothing to plot, measure or "
         "detect.")


def _write_h5(path: Path, **datasets) -> Path:
    with h5py.File(path, "w") as h:
        for name, value in datasets.items():
            h[name] = value
    return path


def _refusal(path: Path, call) -> str:
    with pytest.raises(ReaderError) as exc:
        call(path)
    return str(exc.value)


# ================================================================================================
# 1. the three shapes of one empty acquisition now agree
# ================================================================================================

@pytest.mark.parametrize("layout", [
    pytest.param({"current": np.zeros(0), "voltage": np.zeros(0)}, id="current+voltage"),
    pytest.param({"time": np.zeros(0), "current": np.zeros(0)}, id="time+current"),
    pytest.param({"time": np.zeros(0), "current": np.zeros(0), "voltage": np.zeros(0)},
                 id="time+current+voltage"),
])
@pytest.mark.parametrize("call", [probe_path, open_document],
                         ids=["probe", "open"])
def test_every_zero_sample_layout_gets_the_empty_recording_sentence(tmp_path, layout, call):
    p = _write_h5(tmp_path / "armed.h5", **layout)
    said = _refusal(p, call)
    assert EMPTY in said
    # ...and above all it does not deny a dataset it is about to list.
    assert "no /time base" not in said
    assert "data export" not in said


def test_the_selftests_own_layout_is_the_one_that_used_to_be_called_an_export(tmp_path):
    """``/time`` + ``/current`` is not exotic: NanoScope writes it itself
    (``packaging/nanoscope_selftest.py``'s ``real_slice.h5``)."""
    p = _write_h5(tmp_path / "real_slice.h5",
                  time=np.zeros(0), current=np.zeros(0))
    said = _refusal(p, probe_path)
    assert EMPTY in said
    assert "can be deleted" in said       # the acquisition-armed-and-stopped advice


# ================================================================================================
# 2. what the widened gate must NOT do
# ================================================================================================

def test_a_bare_signal_export_is_still_refused_as_an_export(tmp_path):
    """The gate exists for 'training_split.h5': one signal, no clock, no companion."""
    p = _write_h5(tmp_path / "training_split.h5", current=np.zeros(5_000))
    said = _refusal(p, probe_path)
    assert "no /time base and no second signal channel" in said
    assert "data export, not a recording" in said


def test_a_time_that_is_not_a_channel_at_all_names_what_is_wrong_with_it(tmp_path):
    """A /time this reader cannot read as samples is still a refusal — but the sentence
    may not claim the file has no /time while listing it, and may not call a file that
    carries a clock a data export."""
    p = tmp_path / "string_time.h5"
    with h5py.File(p, "w") as h:
        h["current"] = np.zeros(5_000)
        h.create_dataset("time", data=np.array([b"0.0", b"1.0"]))
    said = _refusal(p, probe_path)
    assert "no /time base" not in said
    assert "data export" not in said
    assert "/time present but unusable as a time base" in said
    assert "which is not a number this reader can turn into a signal" in said
    assert "top-level members: current, time" in said


def test_a_one_sample_clock_beside_one_sample_of_signal_is_a_recording_with_no_rate(tmp_path):
    """Structurally a recording; too short to difference. It opens, and the rate is
    UNKNOWN with the reason said out loud — never fabricated, never silently absent."""
    p = _write_h5(tmp_path / "one.h5", time=np.zeros(1), current=np.full(1, -50.0))
    meta = probe_path(p).meta
    assert meta.n_samples == 1
    assert meta.sample_rate_hz is None
    assert meta.rate_provenance is Provenance.UNKNOWN
    said = " || ".join(meta.issues)
    assert "/time is present in this file but is not usable as a time base" in said
    assert "at least 2 are needed to measure a sample interval" in said


def test_a_real_recording_is_untouched(tmp_path):
    """The no-false-positive half: a well-formed file gains no issue and no note."""
    n = 5_000
    p = _write_h5(tmp_path / "clean.h5",
                  current=np.random.default_rng(0).normal(-50.0, 1.5, n),
                  voltage=np.full(n, -0.6), time=np.arange(n) * 2e-5)
    meta = probe_path(p).meta
    assert meta.issues == ()
    assert meta.n_samples == n
    assert meta.rate_provenance is Provenance.MEASURED
    assert abs(meta.sample_rate_hz - 50_000.0) < 1e-3


def test_a_file_with_no_signal_dataset_still_says_so(tmp_path):
    """A Keras model or an MEArec file: /time alone is not a recording."""
    p = _write_h5(tmp_path / "model.h5", time=np.arange(10) * 2e-5)
    said = _refusal(p, probe_path)
    assert "no readable /current, /voltage or /conductance dataset found" in said


# ================================================================================================
# 3. the seam itself: two questions, two answers
# ================================================================================================

def test_the_structural_and_the_measurable_time_base_are_different_questions(tmp_path):
    from nanoscope.io.hdf5 import Hdf5Reader

    p = _write_h5(tmp_path / "pair.h5", time=np.zeros(0), current=np.zeros(0))
    r = Hdf5Reader(p)
    with h5py.File(p, "r") as h:
        assert r._time_base(h) is not None      # the file carries a clock...
        assert r._time_signal(h) is None        # ...that no rate can be measured from

    q = _write_h5(tmp_path / "good.h5", time=np.arange(10) * 2e-5,
                  current=np.zeros(10))
    with h5py.File(q, "r") as h:
        assert Hdf5Reader(q)._time_base(h) is not None
        assert Hdf5Reader(q)._time_signal(h) is not None


def test_the_refusal_never_denies_a_member_it_lists(tmp_path):
    """The rule the defect broke, stated once: whatever the layout, a refusal that lists
    ``time`` among the members may not also say the file has no /time base."""
    layouts = [
        {"current": np.zeros(0), "time": np.zeros(0)},
        {"current": np.zeros(5_000), "time": np.zeros(0)},
        {"current": np.zeros(5_000), "time": np.zeros(1)},
    ]
    for i, layout in enumerate(layouts):
        p = _write_h5(tmp_path / f"L{i}.h5", **layout)
        try:
            said = " || ".join(probe_path(p).meta.issues)
        except (ReaderError, UnsupportedFormatError) as exc:
            said = str(exc)
        if "members: " in said:
            assert not ("time" in said.split("members: ")[1] and "no /time base" in said), said
