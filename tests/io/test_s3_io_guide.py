"""Third-sweep loader defect IO3-06: the manual described a clock the app does not have.

docs/USER_GUIDE.md §2 promised, of a partial load:

    "Every clock in the app reads the same way: the cursor crosshair, a hand measurement's
     `t_start`, an event's `t_start_s`, and a segment cut from the window."

Three of those four are on the RECORDING clock and are ~1,161 s away from the crosshair on
the owner's own quick ABF. The app carries two clocks ON PURPOSE — ``core/clock.py`` exists
to label them, ``trace_view`` calls its axis "the ONLY surface in the app drawn on the
instrument clock", the event table's column header is ``t_start_s (t₀=0)``, and the shipped
manual has a "Two clocks in one window" paragraph. Confusing the two produced 22.66 pA /
alpha 1.822 where the right window gives 3.99 pA / alpha 0.985 — a 5.7x error in a headline
noise number, on these recordings. The guide was the one surface telling the user they agree.

The guide is a SURFACE, so this test is written the way the other surface-agreement tests
are: it checks the app's behaviour first and the document against it, rather than grepping
prose for its own sake.
"""
from __future__ import annotations

import os

import nanoscope  # noqa: F401  -> Qt plugin-path bootstrap, per tests/conftest.py
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import h5py
import numpy as np
import pytest

from nanoscope.core.clock import CLOCK_SHORT, INSTRUMENT
from nanoscope.core.segment import build_segment_document, derive_segment_uid, persistent_key
from nanoscope.core.types import SampleRange
from nanoscope.io.base import open_document

_GUIDE = Path(__file__).resolve().parents[2] / "docs" / "USER_GUIDE.md"


def _guide_text() -> str:
    return _GUIDE.read_text(encoding="utf-8")


def _windowed_document(tmp_path: Path):
    """A partial load with a non-zero t0 — the situation the guide's paragraph is about."""
    n, dt, t0 = 20_000, 2e-5, 1_154.990912
    p = tmp_path / "window.h5"
    with h5py.File(p, "w") as h:
        h["current"] = np.zeros(n, dtype=np.float64)
        h["voltage"] = np.full(n, -0.6)
        h["time"] = t0 + np.arange(n) * dt
    return open_document(p, SampleRange(6_000, 10_000)), t0 + 6_000 * dt


def test_the_two_clocks_the_guide_now_describes_are_the_ones_the_app_has(tmp_path):
    """The measurement the guide has to match: the document's own clock is the instrument
    clock, and a segment cut from it restarts at 0 while naming its parent instant."""
    doc, window_t0 = _windowed_document(tmp_path)
    assert doc.meta.t0_s == pytest.approx(window_t0, abs=1e-6), (
        "the window carries its own experimental time — the half the guide had right")

    seg = build_segment_document(
        doc, 500, 1_500, name="seg",
        segment_uid=derive_segment_uid(persistent_key(doc), 500, 1_500))
    assert seg.meta.t0_s == 0.0, (
        "a segment is rebased: its own clock starts at 0, which is NOT the crosshair's clock")
    assert seg.origin.t0_abs_s == pytest.approx(window_t0 + 500 * 2e-5, abs=1e-6), (
        "and the instrument-clock instant travels with it, which is what the guide must say")


def test_the_guide_no_longer_claims_every_clock_reads_the_same_way():
    said = _guide_text()
    assert "Every clock in the app reads the same way" not in said, (
        "the false sentence is back in the guide")


def test_the_guide_names_both_clocks_and_where_each_one_appears():
    said = _guide_text()
    assert "Two clocks" in said, "the guide must state that there are two"
    assert CLOCK_SHORT[INSTRUMENT] in said, (
        f"the guide must use the app's own name for the drawn axis' clock "
        f"({CLOCK_SHORT[INSTRUMENT]!r})")
    assert "Recording clock" in said and "first sample that was loaded" in said, (
        "the guide must say where the recording clock's zero is")
    assert "t_start_s" in said, "the event table's column is the surface users trip over"
    assert "segment" in said.lower(), "a segment restarts its own clock and must be named"


def test_the_guide_still_states_the_half_that_was_true():
    """The fix is a correction, not a deletion: a partial load really does carry the window's
    own experimental time, and that paragraph is what tells a user their trace starting at
    600 s is not a bug."""
    said = _guide_text()
    assert "A partial load carries the window's own experimental time" in said
    assert "Whole-file loads are unaffected" in said
