"""ADV4-04 -- a health check that cannot compute its own statistic must say so, not certify.

THE DEFECT. ``baseline_health``'s ``rough_sigma_ratio`` row mapped a NaN ratio to level
'ok' -- the ONE level the Baseline panel renders as PASS/green
(``gui/baseline_panel._CERTIFYING_LEVEL``). The NaN is reachable in exactly one way: the
final detrended noise is 0.00 pA (`_qc_block`'s MAD reads 0 when more than half the
residual sits at exactly one current -- a railed amplifier, a saturated stretch cut into
a segment, quantised data), and the assignment writes NaN instead of dividing by zero.
The user then saw a green PASS row reading

    "rough-pass sigma 0.00 pA against a final noise of 0.00 pA (ratio nan)"

-- a certificate from a check that had measured nothing.

THE INVARIANT RESTORED. A non-finite ``rough_sigma_ratio`` now emits at level 'warn'
with a message that names its own failure ("COULD NOT BE MEASURED: the final detrended
noise is 0.00 pA ..."), so ``BaselineHealth.ok`` is False and the panel renders WARNING,
not PASS. The measurable path is untouched, and its PASS message no longer prints
"nan pA" for an upper MAD it could not take.

THE AUDIT the dossier asked for (the other checks, same hole):
* ``edge_tracking`` -- already emits "COULD NOT BE MEASURED" in words when this same
  noise fails it; its 'ok' is justified there because the length rows condemn first.
* ``overshoot`` -- guarded by ``np.isfinite(over)``, but ``over = n_up/max(n_down,1)``
  is finite for every finite input and the door refuses non-finite traces: unreachable.
* ``mask_reference`` -- ``ref_ratio`` can only be inf if ``mask_frac == 0``, a path
  skipped below the floor; and inf lands on the WARN side of ``> MAX_RATIO`` anyway.
* the scalar block (``state_dwell_s``, ``gating_score``) nullifies non-finite values at
  construction -- display, not verdicts.
None of those certifies on a non-number; ``rough_sigma_ratio`` was the one hole.
"""

from __future__ import annotations

import math
import os

import nanoscope  # noqa: F401  -> Qt bootstrap before any Qt import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis import baseline as B

FS = 50_000.0


def _flat_result():
    """12 s of perfectly flat current -- the railed/saturated shape that makes the
    final detrended noise exactly 0.00 pA and the two-pass ratio NaN."""
    x = np.full(600_000, -800.0)
    return B.estimate_baseline_result(x, FS, polarity=-1)


def test_a_nan_ratio_no_longer_certifies():
    h = _flat_result().health
    c = h.check("rough_sigma_ratio")
    assert c is not None
    assert isinstance(c.value, float) and math.isnan(c.value)
    # The whole defect: this was 'ok'. A check that measured nothing cannot certify.
    assert c.level == "warn"
    assert "COULD NOT BE MEASURED" in c.message
    assert "0.00 pA" in c.message           # it names its own failing input...
    assert "nan" not in c.message.lower()   # ...and never prints NaN at the user.
    # And the block's own verdict follows: a health with an unmeasurable row is not ok.
    assert h.ok is False


def test_the_baseline_panel_renders_warning_not_pass_for_the_nan_row():
    """The shipped surface: the Baseline panel's health block, as rendered
    (``health_text()`` is the characters that reach the screen)."""
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from nanoscope.gui.baseline_panel import BaselinePanel

    panel = BaselinePanel()
    panel.set_health(_flat_result().health, measured_for="detection")
    text = panel.health_text()
    # The row is present, worded as its own failure, and carries the WARNING verdict
    # word (the glyph-bearing word, not just a colour -- the panel's own doctrine).
    assert "rough-pass sigma vs final noise" in text
    assert "COULD NOT BE MEASURED" in text
    line = next(ln for ln in text.splitlines() if "rough-pass sigma vs final noise" in ln)
    assert "WARNING" in line, line
    assert "PASS" not in line, line


def test_the_measurable_path_is_untouched_and_still_passes():
    """A healthy noisy trace: finite ratio, 'ok', and the PASS message prints real
    numbers (no 'nan' anywhere in it)."""
    rng = np.random.default_rng(11)
    n = 600_000
    x = np.full(n, -800.0) + rng.normal(0.0, 5.0, n)
    for s in range(10_000, n - 2_000, 40_000):     # some events, so the mask has work
        x[s:s + 400] += 80.0                        # canonical: events DOWN after pol -1
    h = B.estimate_baseline_result(x, FS, polarity=-1).health
    c = h.check("rough_sigma_ratio")
    assert c is not None and np.isfinite(c.value)
    assert c.level in ("ok", "warn")               # judged by the band, not by finiteness
    assert "nan" not in c.message.lower()
    if c.level == "ok":
        assert "rough-pass sigma" in c.message and "COULD NOT" not in c.message
