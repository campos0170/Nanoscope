"""The manual and the control inventory describe the Signal Conditioning window, not the
in-dock editor it replaced (review D6: both still filed every stage under the Preprocess tab)."""

from __future__ import annotations

import json
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs"


def _controls() -> list[dict]:
    areas = json.loads((DOCS / "inventory.json").read_text(encoding="utf-8"))
    return [c for a in areas for c in a.get("controls", [])]


def test_the_inventory_files_the_stage_controls_under_the_window():
    groups = {c.get("group", "") for c in _controls()}
    stale = sorted(g for g in groups if "Signal chain ▸ " in g or "Signal chain → " in g)
    assert stale == [], stale
    assert {"Signal Conditioning ▸ Low-pass", "Signal Conditioning ▸ Notch",
            "Signal Conditioning ▸ Normalize", "Signal Conditioning ▸ toolbar"} <= groups
    text = json.dumps(_controls(), ensure_ascii=False)
    for name in ("openSignalConditioningButton", "signalConditioningStages",
                 "eventTableNoticeChip", "chainStatusNote", "signalConditioningRevert"):
        assert name in text, name


def test_the_manual_describes_the_window_and_the_protocol_name_rule():
    manual = (DOCS / "manual-markdown" / "manual.md").read_text(encoding="utf-8")
    assert "drops the name back to `— manual —`" not in manual
    assert "On the `Preprocess` tab. Low-pass" not in manual
    assert "Signal Conditioning" in manual and "Ctrl+Shift+F" in manual
