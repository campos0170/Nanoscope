"""Current scientific contracts for real-recording GUI wiring tests.

The original corpus counts predate the versioned automatic masked-baseline
algorithm. These tests exercise the current GUI recipe, not historical replay.
Compare the complete installed table with a fresh calculation outside the GUI;
the analysis tests independently establish the estimators' scientific properties.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis.context import SimpleAnalysisContext
from nanoscope.analysis.modules import DetectSchmitt


def assert_detection_matches_reference(window, expected_windows, source):
    """Require the requested window/source, current policies and exact event table."""
    panel = window._analysis_panel
    actual = panel.current_detection()
    assert actual is not None and actual.module_id == "detect.schmitt"
    assert (actual.doc_id, actual.channel) == window._current_selection
    assert not actual.payload.df.empty
    form = panel.detection_params()
    rough, base = expected_windows
    assert form["baseline_rough_s"] == pytest.approx(rough)
    assert form["baseline_base_s"] == pytest.approx(base)
    assert form["baseline_param_source"] == source
    assert form["noise_policy"] == "auto"
    assert form["k_in"] == 5.0 and form["k_out"] == 2.0
    assert form["min_dwell_s"] == 2e-5 and form["max_dwell_s"] is None
    assert form["filter_cutoff_hz"] is None
    assert form["polarity"] is None
    for stage in ("detrend", "notch", "median", "hp", "rmean", "savgol", "normalize"):
        assert not form[f"{stage}_enabled"], stage

    session = window._session
    doc_id, channel_index = window._current_selection
    document = session.document(doc_id)
    channel = session.calibrated(doc_id, channel_index)
    cfg = B.DEFAULT_BASELINE.replace(rough_s=rough, base_s=base).resolve(channel.sample_rate_hz)
    prov = actual.payload.provenance
    assert prov["baseline_fingerprint"] == cfg.fingerprint()
    assert actual.baseline_fingerprint == cfg.fingerprint()
    assert prov["baseline_param_source"] == source
    assert prov["baseline_sample_count_policy"] == B.SAMPLE_COUNT_POLICY
    assert prov["baseline_mask_noise_policy"] == B.MASK_NOISE_POLICY
    assert prov["noise_policy"] == "auto"
    assert prov["n_samples"] == document.meta.n_samples == channel.data.size

    # Build the expected full-recording context from the selected source and form,
    # never from output-row metadata or a potentially wrong worker's slice bounds.
    expected_params = dict(form)
    expected_params["_calibration_fingerprint"] = tuple(
        session.calibration(doc_id, channel_index).fingerprint())
    expected_params["pore_id"] = document.meta.stem_key
    expected_params.update(baseline_rough_s=rough, baseline_base_s=base,
                           baseline_param_source=source)
    B.clear_baseline_cache()
    reference = DetectSchmitt().run(SimpleAnalysisContext(
        channel, document.meta, expected_params))
    assert not reference.payload.df.empty
    # The GUI installs its curation marker after the analysis module returns. A
    # fresh automatic detection has no hand-added events; every scientific column
    # still has to match the module's result exactly.
    assert set(actual.payload.df) - set(reference.payload.df) == {"added"}
    assert not actual.payload.df["added"].any()
    expected_frame = reference.payload.df.assign(added=False)
    pd.testing.assert_frame_equal(actual.payload.df, expected_frame, check_exact=True)
    pd.testing.assert_frame_equal(panel.current_events_df(), expected_frame, check_exact=True)
    # The timestamp is the only value expected to differ in a fresh computation.
    for key, value in reference.payload.provenance.items():
        if key != "detected_at":
            np.testing.assert_equal(prov[key], value, err_msg=f"provenance {key}")
    return actual.payload.df, prov
