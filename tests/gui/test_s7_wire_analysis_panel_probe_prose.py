"""Sweep 7, S7-JRN1 / MISSED-3 (prose): no stale probe-timing claim survives in the
AnalysisPanel.

The "233 ms / 579 ms" pair in the panel's wiring comment was ``ProbeResult.elapsed_s`` --
the probe STAGE alone -- presented as the cost of the button. Re-measured (agent E, three
presses each, 2026-09-01): the whole job the button dispatches (conditioning + probe +
health second pass) is 1.97 s on the owner's 3 M-sample -600 mV file and 6.38 s on the
9 M-sample -700 mV file, and click-to-proposal wall time is within 60 ms of that. The
panel's prose must carry the measured truth, in the same numbers the tooltip and the
guide now state ("2 s on a 3 M-sample recording and 6 s on 9 M samples").
"""

from __future__ import annotations

import re

from nanoscope.gui import analysis_panel as AP


def _source() -> str:
    return open(AP.__file__, encoding="utf-8").read()


def test_no_stale_probe_stage_number_is_presented_as_the_buttons_cost():
    src = _source()
    for stale in ("~233 ms", "579 ms", "570 ms"):
        assert stale not in src, f"{stale!r} is the probe STAGE, not the button"
    # The one surviving "230 ms" is the docstring's own account of the error it replaced.
    for m in re.finditer(r"230 ms", src):
        line = src[src.rfind("\n", 0, m.start()) + 1: src.find("\n", m.end())]
        assert "earlier" in line or "said" in line, line


def test_the_probe_wiring_comment_states_the_measured_seconds():
    src = _source()
    i = src.index("self._baseline.probeRequested.connect(self._on_probe_requested)")
    comment = src[max(0, i - 800): i]
    assert "2 s" in comment and "3 M-sample" in comment, comment
    assert "6 s" in comment and "9 M" in comment, comment
    assert "_on_probe_requested" in comment, "the comment must point at the measurement"
