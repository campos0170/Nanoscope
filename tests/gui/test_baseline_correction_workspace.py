"""Focused contract tests for the baseline-correction presentation workspace."""

from __future__ import annotations

# Qt bootstrap must precede any Qt import.
import os

import nanoscope  # noqa: F401,E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from nanoscope.gui.baseline_panel import BaselinePanel  # noqa: E402
from nanoscope.gui.baseline_workspace import (  # noqa: E402
    DISPLAY_ZERO,
    RUNTIME_RETAIN_OFFSET,
    BaselineCorrectionWorkspace,
)


@pytest.fixture
def workspace(qapp, qtbot):
    panel = BaselinePanel()
    widget = BaselineCorrectionWorkspace(
        panel,
        method_choices=(("block median", "block_median"),
                        ("event-masked median", "masked_median")),
    )
    qtbot.addWidget(widget)
    return widget, panel


def test_workspace_hosts_the_existing_open_pore_panel(workspace):
    widget, panel = workspace
    assert widget.baseline_panel is panel
    assert panel.parent() is widget
    assert widget.output_mode() == DISPLAY_ZERO
    assert "display-only" in widget.contract_label.text()


def test_workspace_does_not_floor_the_inspector_dock_width(workspace):
    """Long controls must not make the trace viewer permanently hard to widen."""
    widget, _panel = workspace
    assert widget.minimumSizeHint().width() < 300


def test_user_intents_are_emitted_but_runtime_state_sync_is_silent(workspace, qtbot):
    widget, _panel = workspace

    with qtbot.assertNotEmitted(widget.runtimeEnabledRequested), \
            qtbot.assertNotEmitted(widget.runtimeMethodRequested):
        widget.sync_runtime({"detrend_enabled": True,
                             "detrend_method": "masked_median"})
    assert widget.runtime_enabled.isChecked()
    assert widget.runtime_method.currentData() == "masked_median"

    with qtbot.waitSignal(widget.modeChanged) as mode:
        widget.output_mode_combo.setCurrentIndex(1)
    assert mode.args == [RUNTIME_RETAIN_OFFSET]

    widget.runtime_enabled.setChecked(False)
    with qtbot.waitSignal(widget.runtimeEnabledRequested) as enabled:
        widget.runtime_enabled.setChecked(True)
    assert enabled.args == [True]

    with qtbot.waitSignal(widget.runtimeMethodRequested) as method:
        widget.runtime_method.setCurrentIndex(0)
    assert method.args == ["block_median"]


def test_preview_mapping_stays_aligned_and_is_capped_for_drawing(workspace):
    widget, _panel = workspace
    n = 10_001
    time_s = np.linspace(0.0, 1.0, n)
    raw = np.sin(time_s)
    baseline = raw * 0.1
    corrected = raw - baseline

    widget.set_preview({
        "time_s": time_s,
        "raw_current": raw,
        "fitted_baseline": baseline,
        "corrected_current": corrected,
        "mode": RUNTIME_RETAIN_OFFSET,
        "units": "pA",
    })

    assert widget._preview_time_s is not None
    assert widget._preview_time_s.size == 8_000
    assert widget._preview_time_s[0] == pytest.approx(time_s[0])
    assert widget._preview_time_s[-1] == pytest.approx(time_s[-1])
    x_raw, y_raw = widget.raw_curve.getData()
    x_base, y_base = widget.baseline_curve.getData()
    x_out, y_out = widget.corrected_curve.getData()
    assert x_raw.size == y_raw.size == x_base.size == y_base.size == x_out.size == y_out.size == 8_000
    assert "runtime-style preview" in widget.preview_status.text()


def test_clear_preview_and_busy_state_do_not_disable_explanatory_buttons(workspace):
    widget, _panel = workspace
    widget.set_busy(True)
    assert widget.is_busy()
    assert widget.refresh_button.isEnabled()
    assert widget.export_button.isEnabled()

    widget.clear_preview("No usable current channel is selected.")
    assert not widget.is_busy()
    assert "No usable current channel" in widget.preview_status.text()
    assert widget.raw_curve.getData() == (None, None)
