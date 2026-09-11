"""Presentation-only decimation and a preview with no commit action."""
import os

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.open_pore_preview import OpenPorePreviewDialog, decimate_reference_trace


def payload(n_events=2):
    time = np.arange(1000) / 1000
    current = 100 + np.sin(time * 50)
    current[200:250] -= 30
    return {
        "time_s": time, "current_pA": current, "reference_pA": np.full(1000, 100.),
        "event_start_s": np.arange(n_events) * .1 + .1,
        "event_end_s": np.arange(n_events) * .1 + .15,
        "event_level_pA": np.full(n_events, 70.), "event_count": n_events,
        "polarity": -1, "noise_pA": .7,
        "warnings": ["UNIQUE_ASSUMPTION: user-supplied reference is not independently verified."],
        "reference_details": {"construction": "supplied_constant", "user_assertion_not_verified": True},
        "title": "Reference preview — recording A",
    }


def test_short_trace_is_exact_aligned_copy():
    t = np.arange(10.)
    y, b = t * 3, t * -2
    result = decimate_reference_trace(t, y, b)
    assert set(result) == {"time_s", "current_pA", "reference_pA"}
    for actual, expected in zip(result.values(), (t, y, b)):
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, expected)


@pytest.mark.parametrize("budget", [6, 7, 22, 100, 20000])
def test_bin_extrema_and_endpoints_survive_within_point_budget(budget):
    n = 30000
    t = np.arange(n) / 1000
    rng = np.random.default_rng(312)
    current, reference = rng.normal(size=(2, n))
    result = decimate_reference_trace(t, current, reference, max_points=budget)
    selected = np.searchsorted(t, result["time_s"])
    assert selected.size <= budget
    assert selected[0] == 0 and selected[-1] == n - 1
    assert np.all(np.diff(selected) > 0)
    np.testing.assert_array_equal(result["current_pA"], current[selected])
    np.testing.assert_array_equal(result["reference_pA"], reference[selected])
    edges = np.linspace(0, n, (budget - 2) // 4 + 1, dtype=np.int64)
    kept = set(selected)
    for a, b in zip(edges[:-1], edges[1:]):
        for values in (current, reference):
            assert a + np.argmin(values[a:b]) in kept
            assert a + np.argmax(values[a:b]) in kept


@pytest.mark.parametrize("args,kwargs", [
    (([0, 1], [1], [1, 1]), {}),
    (([[0, 1]], [[1, 1]], [[1, 1]]), {}),
    (([0, 1], [np.nan, 1], [1, 1]), {}),
    (([1, 0], [1, 1], [1, 1]), {}),
    (([0, 0], [1, 1], [1, 1]), {}),
    (([0, 1], [1, 1], [1, 1]), {"max_points": 5}),
    (([0, 1], [1, 1], [1, 1]), {"max_points": True}),
    (([0, 1], [1, 1], [1, 1]), {"max_points": 10.5}),
])
def test_bad_display_arrays_or_budget_are_rejected(args, kwargs):
    with pytest.raises(ValueError):
        decimate_reference_trace(*args, **kwargs)


def test_preview_is_modeless_has_no_apply_and_keeps_warnings_off_plot(qapp, qtbot):
    data = payload()
    original = data["current_pA"].copy()
    dialog = OpenPorePreviewDialog(data)
    qtbot.addWidget(dialog)
    dialog.show()
    qapp.processEvents()
    assert not dialog.isModal()
    assert dialog.windowModality() == QtCore.Qt.WindowModality.NonModal
    assert "existing analysis unchanged" in dialog.notice_label.text()
    assert "press Detect" in dialog.notice_label.text()
    assert dialog.buttons.standardButtons() == QtWidgets.QDialogButtonBox.StandardButton.Close
    assert len(dialog.plot.listDataItems()) == 3
    assert dialog.plot.getPlotItem().titleLabel.text == ""
    assert not dialog.details.isVisible()
    assert "UNIQUE_ASSUMPTION" in dialog.details.toPlainText()
    dialog.details_toggle.click()
    assert dialog.details.isVisible()
    assert dialog.details.isReadOnly()
    dialog.details_toggle.click()
    assert not dialog.details.isVisible()
    np.testing.assert_array_equal(data["current_pA"], original)
    assert not hasattr(dialog, "applyRequested")
    assert not hasattr(dialog, "result_params")


def test_many_candidates_use_one_scatter_item(qapp, qtbot):
    dialog = OpenPorePreviewDialog(payload(12000))
    qtbot.addWidget(dialog)
    x, y = dialog.candidate_markers.getData()
    assert len(x) == len(y) == 12000
    assert len(dialog.plot.listDataItems()) == 3
    assert "12,000 candidate events" in dialog.summary_label.text()


def test_unplottable_levels_are_reported_without_changing_candidate_count(qapp, qtbot):
    data = payload()
    data["event_level_pA"][0] = np.nan
    dialog = OpenPorePreviewDialog(data)
    qtbot.addWidget(dialog)
    assert "2 candidate events" in dialog.summary_label.text()
    assert "1 candidate has no finite display marker" in dialog.summary_label.text()
    assert len(dialog.candidate_markers.getData()[0]) == 1


def test_mismatched_candidate_arrays_refuse(qapp):
    with pytest.raises(ValueError):
        OpenPorePreviewDialog(payload() | {"event_count": 3})
    with pytest.raises(ValueError):
        OpenPorePreviewDialog(payload() | {"event_end_s": np.array([])})
