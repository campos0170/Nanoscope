"""Malformed user references disable amplitude curation without crashing restore."""
import copy

import pandas as pd
import pytest

from nanoscope.analysis.baseline import BaselineConfig
from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from .test_event_amplitude_curation import builders


KEY = "baseline_open_pore_reference"
RECIPE = {"version": 1, "mode": "constant", "current_pA": 100.,
          "intervals_s": [], "interpolation": "constant", "notes": "known current"}


@pytest.fixture
def referenced(qapp, qtbot, tmp_path):
    session, doc_id, channel = builders._session_with_doc(tmp_path)
    ch = session.calibrated(doc_id, channel)
    df, prov = detect_events(ch.data, ch.sample_rate_hz, min_dwell_s=5e-4,
                             baseline_config=BaselineConfig(open_pore_reference=RECIPE))
    inspector = builders._inspector(qtbot, session)
    return inspector, doc_id, channel, EventSet(df, prov)


@pytest.mark.parametrize("change", [
    {KEY: {}}, {KEY: None}, {"remove_recipe": True},
    {"baseline_fingerprint": None}, {"normalize_enabled": True},
    {"hp_enabled": True}, {"detrend_enabled": True}, {"units": "nA"},
])
def test_restore_and_note_only_refresh_refuse_bad_reference_without_row_writes(referenced, change):
    inspector, doc_id, channel, events = referenced
    provenance = copy.deepcopy(events.provenance)
    if change.get("remove_recipe"):
        provenance.pop(KEY)
    else:
        provenance.update(change)
    before = events.df.copy(deep=True)
    inspector.set_events(doc_id, channel, events.df, provenance=provenance)
    controls = inspector._amplitude
    assert controls._error
    assert "Amplitude unavailable:" in controls.note.text()
    assert not controls.apply_button.isEnabled()
    assert not controls.edit_level.isEnabled()
    assert not inspector._level_line.movable
    assert not controls.apply_current(method="manual", manual_level=75.)
    pd.testing.assert_frame_equal(events.df, before, check_exact=True)

    # A curation/status refresh can reach sync without a full trace render. It
    # must catch the resolver exception too, not crash from the note builder.
    controls._error = ""
    controls.sync(events.df.iloc[0], force=True)
    assert controls._error
    assert not controls.apply_button.isEnabled()
    assert not inspector._level_line.movable
    assert "Amplitude unavailable:" in controls.note.text()
    pd.testing.assert_frame_equal(events.df, before, check_exact=True)


def test_valid_reference_manual_amplitude_unchanged_and_recovers_after_bad_restore(referenced):
    inspector, doc_id, channel, events = referenced
    malformed = dict(events.provenance) | {KEY: {}}
    inspector.set_events(doc_id, channel, events.df, provenance=malformed)
    assert inspector._amplitude._error
    inspector.set_events(doc_id, channel, events.df, provenance=dict(events.provenance))
    controls = inspector._amplitude
    assert not controls._error
    assert controls.apply_button.isEnabled()
    assert "Legacy reference" not in controls.note.text()
    before = events.df.copy(deep=True)
    assert controls.apply_current(method="manual", manual_level=75.)
    assert events.df.iloc[0].amplitude_pA == pytest.approx(25.)
    for column in before.columns:
        pd.testing.assert_series_equal(events.df[column], before[column], check_exact=True)


def test_current_measurement_units_cannot_override_a_saved_pa_reference_silently(referenced):
    inspector, doc_id, channel, events = referenced
    inspector.set_events(doc_id, channel, events.df, provenance=dict(events.provenance))
    inspector._units = "nA"
    before = events.df.copy(deep=True)
    inspector._amplitude.sync(events.df.iloc[0], force=True)
    assert "pA" in inspector._amplitude.note.text()
    assert not inspector._amplitude.apply_button.isEnabled()
    assert not inspector._amplitude.apply_current(method="manual", manual_level=75.)
    pd.testing.assert_frame_equal(events.df, before, check_exact=True)
