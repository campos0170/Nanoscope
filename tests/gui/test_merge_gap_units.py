"""The Detection tab's merge gap speaks MILLISECONDS, like the Merge panel.

The two merge windows in this app used to disagree about units: MergePanel's spin is
``window_ms`` (" ms", 2 decimals) while DetectionPanel's was ``merge_gap_s`` (" s",
7 decimals), so a user wanting 15 ms typed "15" on one panel and "0.015" on the other.
That is not hypothetical pedantry — merge_panel.py's own docstring records the defect it
already caused: the first hands-on replication of the tutorial typed 15 into a µs-era
field meaning 15 ms and merged at 15 µs, a 1000x silent under-merge, and the user
reasonably concluded merging "was not giving good results". The unit fix reached only one
of the two panels; these tests pin the other one.

What is pinned here:

* the WIDGET is ``merge_gap_ms`` and reads milliseconds (suffix, range, no seconds-named
  alias left behind for a caller to set in the wrong unit);
* the params() CONTRACT is unchanged — the key is still ``merge_gap_s``, still SECONDS,
  because that is detect_events' kwarg and what every saved project/protocol stores;
* the ms -> s / s -> ms legs round-trip losslessly, including the MIGRATION case: a dict
  written before this change carries seconds and must land on the form as milliseconds
  (0.015 s -> 15 ms), never be reinterpreted as 0.015 ms;
* 0 is still the off-sentinel here (params() -> None), which is the ONE thing the two
  panels are deliberately allowed to disagree about: a detection run can skip merging,
  while "Merge all" is a button whose only meaning is merge.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from nanoscope.gui.merge_panel import MergePanel
from nanoscope.gui.panels import DetectionPanel


def test_merge_gap_widget_is_millisecond_named_and_suffixed(qtbot):
    """The widget's NAME and its suffix both say ms, and no seconds alias survives.

    The name matters as much as the suffix: `panel.merge_gap_s.setValue(15)` from any
    caller or test would have been a 1000x error with no visible symptom, so the
    seconds-named attribute must be gone rather than kept as a courtesy alias.
    """
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert " ms" in panel.merge_gap_ms.suffix()
    assert not hasattr(panel, "merge_gap_s")
    # Range scaled from the old 0-1 s, so no saved value became unrepresentable.
    assert panel.merge_gap_ms.minimum() == 0.0
    assert panel.merge_gap_ms.maximum() == pytest.approx(1000.0)
    # 4 decimals of ms == the old 7 decimals of seconds (1e-7 s either way).
    assert panel.merge_gap_ms.decimals() == 4


def test_typed_15_means_15_ms_on_both_panels(qtbot):
    """The regression the whole change exists for: "15" is 15 ms everywhere."""
    det = DetectionPanel()
    qtbot.addWidget(det)
    merge = MergePanel()
    qtbot.addWidget(merge)

    det.merge_gap_ms.setValue(15.0)
    merge.window_ms.setValue(15.0)

    assert det.params()["merge_gap_s"] == pytest.approx(0.015)
    assert det.params()["merge_gap_s"] == pytest.approx(merge.window_s())


def test_params_emits_seconds_under_the_unchanged_key(qtbot):
    """params() is a detect_events kwargs dict: the key and its unit did NOT change."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.merge_gap_ms.setValue(0.15)  # the Merge panel's default window
    p = panel.params()
    assert "merge_gap_ms" not in p
    assert p["merge_gap_s"] == pytest.approx(1.5e-4)


@pytest.mark.parametrize("ms", [0.01, 0.15, 1.0, 10.0, 15.0, 123.4567, 1000.0])
def test_params_set_params_round_trip_is_lossless(qtbot, ms):
    """Widget -> params() -> set_params() -> widget, for a fresh panel each way.

    A double conversion through 1e-3/1e3 is where a unit change usually leaks precision,
    so this walks the whole span the spin can hold, including its maximum.
    """
    src = DetectionPanel()
    qtbot.addWidget(src)
    src.merge_gap_ms.setValue(ms)
    stored = src.params()

    dst = DetectionPanel()
    qtbot.addWidget(dst)
    dst.set_params(stored)
    assert dst.merge_gap_ms.value() == pytest.approx(ms)
    assert dst.params()["merge_gap_s"] == pytest.approx(stored["merge_gap_s"])


def test_legacy_seconds_params_restore_as_milliseconds(qtbot):
    """MIGRATION: projects, protocols and restored sessions predate the unit change.

    They stored ``merge_gap_s`` in SECONDS under this same key (protocols are QSettings
    JSON captured straight out of params()). Because params() still emits seconds, such a
    file is not "legacy" in format at all — but only if set_params() applies the s -> ms
    leg. Reinterpreting a stored 0.015 s as 0.015 ms would be the same 1000x under-merge
    the ms change was made to kill, silently, on every reopened project.
    """
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    panel.set_params({"merge_gap_s": 0.015})
    assert panel.merge_gap_ms.value() == pytest.approx(15.0)
    assert panel.params()["merge_gap_s"] == pytest.approx(0.015)


def test_zero_is_still_off_in_both_directions(qtbot):
    """0 -> None -> 0. The one deliberate difference from the Merge panel."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    assert panel.merge_gap_ms.value() == 0.0
    assert panel.params()["merge_gap_s"] is None

    panel.merge_gap_ms.setValue(5.0)
    panel.set_params({"merge_gap_s": None})
    assert panel.merge_gap_ms.value() == 0.0
    assert panel.params()["merge_gap_s"] is None

    # The sentinel is spelled out on the control, not just in the docstring.
    assert "0 = off" in panel.merge_gap_ms.suffix()
    # ...and the Merge panel is NOT allowed to grow the same sentinel: its floor is a
    # real minimum, because a merge button with a "don't merge" window is a dead control.
    merge = MergePanel()
    qtbot.addWidget(merge)
    assert merge.window_ms.minimum() > 0.0


def test_merge_gap_has_a_tooltip_covering_off_and_refusal(qtbot):
    """It had none at all, unlike its ``max_dwell_s`` sibling two rows above.

    The occupancy sentence is the one the user will actually need: a run whose merged
    span would be mostly open pore is refused, so an expected merge that does not happen
    reads as a deliberate refusal instead of a broken window.
    """
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    tip = panel.merge_gap_ms.toolTip()
    assert tip
    assert "0 turns the merge stage off" in tip
    assert "open pore" in tip
    assert "REFUSED" in tip


def test_merge_gap_edits_still_re_emit_params(qtbot):
    """The rename had to reach the paramsChanged wiring list too, not just params()."""
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    seen: list[dict] = []
    panel.paramsChanged.connect(seen.append)
    panel.merge_gap_ms.setValue(2.0)
    assert seen and seen[-1]["merge_gap_s"] == pytest.approx(0.002)
