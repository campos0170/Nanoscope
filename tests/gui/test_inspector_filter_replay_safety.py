"""Failed saved conditioning cannot overwrite event geometry on a raw fallback."""
import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

import nanoscope.gui.event_inspector as inspector_module
from .test_event_amplitude_curation import builders


@pytest.fixture
def replay(qapp, qtbot, tmp_path):
    session, doc_id, channel = builders._session_with_doc(tmp_path)
    events = builders._events_for(session, doc_id, channel)
    inspector = builders._inspector(qtbot, session)
    inspector.set_events(doc_id, channel, events.df, provenance=dict(events.provenance))
    raw = np.asarray(session.calibrated(doc_id, channel).data)
    return inspector, session, doc_id, events, raw


@pytest.mark.parametrize("cutoff", [0., -1., np.nan, np.inf, 50_000., "invalid"])
def test_saved_invalid_lowpass_cannot_be_replayed_as_unfiltered(replay, cutoff):
    inspector, _, _, _, raw = replay
    # Rendering already cached the valid saved recipe's trace. A refused replay must
    # leave that entry exactly as it was and put nothing in its place.
    saved = inspector._cond_cache
    inspector._provenance = dict(inspector._provenance, filter_cutoff_hz=cutoff)
    with pytest.raises(ValueError, match="saved low-pass"):
        inspector._conditioned(raw, builders.FS)
    assert inspector._cond_cache is saved


def test_replay_failure_does_not_cache_raw_and_valid_cache_survives(replay, monkeypatch):
    inspector, _, _, _, raw = replay
    original = inspector_module.detection_signal
    first = inspector._conditioned(raw, builders.FS)
    saved_cache = inspector._cond_cache
    inspector._provenance = dict(inspector._provenance, filter_cutoff_hz=2000.)
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("filter replay failed")

    monkeypatch.setattr(inspector_module, "detection_signal", fail)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="filter replay failed"):
            inspector._conditioned(raw, builders.FS)
        assert inspector._cond_cache is saved_cache
    assert len(calls) == 2
    monkeypatch.setattr(inspector_module, "detection_signal", original)
    filtered = inspector._conditioned(raw, builders.FS)
    assert filtered is not first
    assert not np.array_equal(filtered, raw)
    assert inspector._conditioned(raw, builders.FS) is filtered


@pytest.mark.parametrize("kind", ["exception", "invalid_cutoff"])
def test_failed_drag_preserves_rows_optional_amplitude_history_signal_handles_and_view(replay, monkeypatch, kind):
    inspector, session, doc_id, events, raw = replay
    controls = inspector._amplitude
    assert controls.apply_current(method="manual", manual_level=75.)
    events.df.loc[events.df.index[0], "accepted"] = False
    events.df.loc[events.df.index[0], "flag"] = "user_reject"
    before = events.df.copy(deep=True)
    history = []
    inspector.eventEdited.connect(history.append)
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args))
    # Rendering already cached the conditioned trace, and a drag would reuse it. Start
    # empty so the drag has to replay the saved filter, and so the closing ``is None``
    # check shows that a failed replay cached nothing.
    inspector._cond_cache = None
    if kind == "exception":
        def fail(*args, **kwargs):
            raise ValueError("saved filter could not run")
        monkeypatch.setattr(inspector_module, "detection_signal", fail)
    else:
        inspector._provenance = dict(inspector._provenance, filter_cutoff_hz=50_000.)
    old_range = np.asarray(inspector._plot.getViewBox().viewRange())
    old_handles = inspector._region.getRegion()
    old_curve = tuple(np.array(x, copy=True) for x in inspector._curve.getData())
    row = events.df.iloc[0]
    xs = session.x_axis(doc_id).values
    inspector._syncing = True
    try:
        inspector._region.setRegion((xs[int(row.start_idx) + 2], xs[int(row.end_idx) - 2]))
    finally:
        inspector._syncing = False
    inspector._on_region_edited()
    pd.testing.assert_frame_equal(events.df, before, check_exact=True)
    assert history == []
    assert len(warnings) == 1
    assert warnings[0][1] == "Event bounds unchanged"
    assert "No event bounds, measurements or curation were changed" in warnings[0][2]
    assert inspector._region.getRegion() == old_handles
    np.testing.assert_array_equal(inspector._plot.getViewBox().viewRange(), old_range)
    for new, old in zip(inspector._curve.getData(), old_curve):
        np.testing.assert_array_equal(new, old)
    assert inspector._cond_cache is None
    # The inspection view remains usable even though measurement cannot replay.
    inspector.next_event()
    assert inspector._current_id == int(events.df.iloc[1].event_id)
    assert inspector._curve.isVisible()


def test_successful_edit_still_remeasures_and_emits_exactly_one_history_signal(replay):
    inspector, _, _, events, raw = replay
    row = events.df.iloc[0]
    edits = []
    inspector.eventEdited.connect(edits.append)
    inspector._apply_edited_bounds(0, int(row.start_idx) + 2, int(row.end_idx) - 2, raw)
    assert edits == [int(row.event_id)]
    assert int(events.df.iloc[0].start_idx) == int(row.start_idx) + 2
    assert int(events.df.iloc[0].end_idx) == int(row.end_idx) - 2
