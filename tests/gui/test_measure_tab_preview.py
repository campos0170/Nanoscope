"""The Measure tab shows the event Ctrl+M would store (docs/MANUAL_EVENT_TABLE_DESIGN.md, section 8).

A live preview line measured by the add door's own row measurement, unstored statistics against the
target table's baseline, noise and signal, no *measure on* choice when an owner supplies a basis,
and the old measurement list shown only while it holds something. Wired through a real MainWindow.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import (
    ANALYSED_START_KEY,
    ANALYSED_STOP_KEY,
    MANUAL_MODULE_ID,
    EventSet,
)
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.measure_panel import MeasurePanel

FS = 50_000.0
N = 60_000
I0 = -1000.0
DEPTH = 300.0
_DETECTED = [(10_000, 10_300), (30_000, 30_300)]
_MISSED = (20_000, 20_300)


def _signal(spans, *, n=N, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, I0, dtype=np.float64) + rng.normal(0.0, 8.0, n)
    for s, e in spans:
        x[s:e] += DEPTH
    return x.astype(np.float32)


def _doc(tmp_path, signal, *, rate=FS, name="preview.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=rate, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


def _open(win, tmp_path, *, detect=True, rate=FS, spans=None):
    signal = _signal(spans if spans is not None else _DETECTED + [_MISSED])
    doc_id = win._session.add_document(_doc(tmp_path, signal, rate=rate))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    if detect:
        df, prov = detect_events(_signal(_DETECTED, seed=1), FS, baseline=I0, min_dwell_s=1e-4)
        win._session.add_result(AnalysisResult(
            module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
            params={}, calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
            payload=EventSet(df, prov)))
    return doc_id


def _bracket(win, doc_id, span=_MISSED):
    mp = win._analysis_panel.measure_panel
    mp.set_active(True)
    xs = win._session.x_axis(doc_id).values
    win._trace_view.set_measure_region(float(xs[span[0]]), float(xs[span[1]]))
    return mp


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _stored(win, doc_id, module_id):
    return win._session.results.get(doc_id, 0, module_id).payload.df


# ------------------------------------------------------------------------------------------------
# the preview IS the row that gets stored
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("detect, table, module_id", [
    (False, "Manual", MANUAL_MODULE_ID),
    (True, "Schmitt", "detect.schmitt"),
])
def test_the_preview_row_is_the_row_the_add_stores(win, tmp_path, detect, table, module_id):
    doc_id = _open(win, tmp_path, detect=detect)
    mp = _bracket(win, doc_id)

    preview = mp.event_preview()
    assert preview is not None and preview.table == table and preview.row is not None
    assert not preview.reason and not preview.refusal
    text = mp.event_preview_text()
    assert text.startswith(f"→ {table}: dwell ") and "depth " in text and "baseline " in text

    mp._add_button.click()
    df = _stored(win, doc_id, module_id)
    row = df.loc[(df["start_idx"] == _MISSED[0]) & (df["end_idx"] == _MISSED[1])].iloc[0]
    for key in ("dwell_s", "depth_pA", "baseline_pA", "frac_depth", "t_start_s"):
        assert float(row[key]) == pytest.approx(float(preview.row[key]), rel=1e-9), key


def test_the_preview_names_the_basis_and_the_hand_made_counts(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    mp = _bracket(win, doc_id)
    lines = mp.event_preview_text().splitlines()
    assert lines[1] == "Measured on the raw signal. No hand-made events yet."

    mp._add_button.click()
    _bracket(win, doc_id, (12_000, 12_300))
    assert mp.event_preview_text().splitlines()[1] == (
        "Measured on the raw signal. Hand-made here: Manual 1.")


def test_with_no_region_the_preview_says_how_to_make_one(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    mp = win._analysis_panel.measure_panel
    assert "press C over an event" in mp.event_preview_text()
    assert mp.event_preview() is None
    _bracket(win, doc_id)
    assert mp.event_preview() is not None
    mp.set_active(False)
    assert mp.event_preview() is None and "press C over an event" in mp.event_preview_text()


# ------------------------------------------------------------------------------------------------
# refusals show before the click
# ------------------------------------------------------------------------------------------------
def test_an_overlap_shows_in_the_preview_and_the_door_still_refuses(win, tmp_path):
    doc_id = _open(win, tmp_path)
    mp = _bracket(win, doc_id, (10_100, 10_200))
    preview = mp.event_preview()
    assert preview.row is None and "overlaps event" in preview.refusal
    assert mp.event_preview_text().startswith("Would not be added to Schmitt: That span overlaps")
    assert mp._add_button.isEnabled()                      # the door explains; the button is not mute
    n = len(_stored(win, doc_id, "detect.schmitt"))
    mp._add_button.click()
    assert len(_stored(win, doc_id, "detect.schmitt")) == n
    assert "overlaps event" in win._analysis_panel.last_missed_refusal


def test_a_span_outside_what_the_detector_searched_is_previewed_as_refused(win, tmp_path):
    doc_id = _open(win, tmp_path)
    res = win._session.results.get(doc_id, 0, "detect.schmitt")
    prov = dict(res.payload.provenance)
    prov[ANALYSED_START_KEY], prov[ANALYSED_STOP_KEY] = 0, 15_000
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0, params={},
        calibration_fingerprint=res.calibration_fingerprint, payload=EventSet(res.payload.df, prov)))
    mp = _bracket(win, doc_id)
    assert "outside the stretch Schmitt searched" in mp.event_preview().refusal


def test_without_a_sample_rate_the_preview_says_why_and_the_readout_still_works(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False, rate=None)
    mp = _bracket(win, doc_id)
    preview = mp.event_preview()
    assert preview is not None and preview.reason and "sample rate" in preview.reason
    assert mp.current_measurement() is not None            # local measurement, as before
    assert mp.event_preview_text().startswith("No event can be stored here:")


# ------------------------------------------------------------------------------------------------
# the unstored statistics answer against the table's baseline and signal
# ------------------------------------------------------------------------------------------------
def test_the_statistics_use_the_tables_baseline_and_noise(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    mp = _bracket(win, doc_id)
    preview = mp.event_preview()
    stats = mp.current_measurement()
    assert stats["baseline"] == pytest.approx(preview.baseline, rel=1e-9)
    assert stats["baseline"] == pytest.approx(float(preview.row["baseline_pA"]), rel=1e-6)
    assert stats["amplitude"] > 0                           # an event, in the table's own sense
    assert stats["amplitude"] == pytest.approx(DEPTH, rel=0.1)
    assert stats["snr"] == pytest.approx(stats["amplitude"] / preview.noise_rms, rel=0.05)


def test_the_measure_on_choice_goes_away_when_the_owner_supplies_the_basis(win, tmp_path):
    _open(win, tmp_path, detect=False)
    mp = win._analysis_panel.measure_panel
    assert mp._basis_combo.isHidden() and mp._basis_note.isHidden()


def test_a_panel_on_its_own_keeps_the_choice_and_its_old_measurement(qapp, qtbot, tmp_path):
    from nanoscope.core.session import Session
    from nanoscope.gui.trace_view import TraceView

    session = Session()
    view = TraceView(session)
    panel = MeasurePanel(session, view)
    qtbot.addWidget(panel)
    assert not panel._basis_combo.isHidden() or not panel.isVisible()
    assert panel.event_preview_text() == "" and panel.event_preview() is None


# ------------------------------------------------------------------------------------------------
# long spans, table switches, the old list
# ------------------------------------------------------------------------------------------------
def test_a_long_span_is_not_previewed_live_but_is_still_added_exactly(win, tmp_path):
    n = 600_000
    signal = np.full(n, I0, dtype=np.float32)
    signal += np.random.default_rng(3).normal(0, 8, n).astype(np.float32)
    signal[100_000:400_000] += DEPTH                        # a 300k-sample block: above the bound
    doc_id = win._session.add_document(_doc(tmp_path, signal, name="long.dat"))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    mp = _bracket(win, doc_id, (100_000, 400_000))
    preview = mp.event_preview()
    assert preview is not None and preview.row is None and not preview.refusal
    assert "too long to preview live" in mp.event_preview_text()
    assert mp.current_measurement() is not None
    mp._add_button.click()
    df = _stored(win, doc_id, MANUAL_MODULE_ID)
    assert (int(df["start_idx"].iloc[0]), int(df["end_idx"].iloc[0])) == (100_000, 400_000)


def test_choosing_another_table_moves_the_preview_without_moving_the_mask(win, tmp_path):
    doc_id = _open(win, tmp_path)
    panel = win._analysis_panel
    mp = _bracket(win, doc_id)
    assert mp.event_preview().table == "Schmitt"
    bounds = win._trace_view.measure_bounds()

    selector = panel._result_selector
    selector.setCurrentIndex(selector.findData(MANUAL_MODULE_ID))
    win._trace_keys._paint_label()                          # the debounce, without waiting for it
    assert mp.event_preview().table == "Manual"
    assert win._trace_view.measure_bounds() == bounds


def test_the_old_list_shows_only_while_it_holds_something(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    mp = _bracket(win, doc_id)
    assert mp._legacy_box.isHidden() and mp._legacy_box.title() == "Old measurements (0)"
    assert mp.add_current() is True                         # the method stays; the UI no longer offers it
    assert not mp._legacy_box.isHidden() and mp._legacy_box.title() == "Old measurements (1)"
    assert mp._update_button.isHidden()
    mp.set_measurements_df(None)
    assert mp._legacy_box.isHidden()


def test_a_legacy_row_is_stamped_with_the_basis_the_numbers_were_measured_on(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    mp = _bracket(win, doc_id)
    assert mp.add_current() is True
    assert str(mp.measurements_df["signal_basis"].iloc[0]) == "raw"


# ------------------------------------------------------------------------------------------------
# a processed table is measured on its own processed signal
# ------------------------------------------------------------------------------------------------
def test_a_table_with_a_low_pass_previews_and_measures_on_its_processed_signal(win, tmp_path):
    from nanoscope.analysis.measure import measure_selection
    from nanoscope.analysis.modules import detection_signal

    signal = _signal(_DETECTED + [_MISSED])
    doc_id = win._session.add_document(_doc(tmp_path, signal))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    df, prov = detect_events(_signal(_DETECTED, seed=1), FS, baseline=I0, min_dwell_s=1e-4,
                             filter_cutoff_hz=5000.0)
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0, params={},
        calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))

    mp = _bracket(win, doc_id)
    preview = mp.event_preview()
    assert preview.basis_kind == "processed" and "low-pass" in preview.basis_text
    assert mp.event_preview_text().splitlines()[1].startswith(
        "Measured on Schmitt's signal (low-pass bessel 5000 Hz")
    raw = measure_selection(signal, FS, *_MISSED)
    want = measure_selection(detection_signal(signal, FS, prov), FS, *_MISSED)
    got = mp.current_measurement()["std"]
    assert got == pytest.approx(want["std"], rel=1e-6)              # the table's own signal...
    assert got != pytest.approx(raw["std"], rel=0.05)               # ...and not the raw one
    assert mp._measured_basis_kind == "processed"

    mp._add_button.click()
    stored = _stored(win, doc_id, "detect.schmitt")
    row = stored.loc[(stored["start_idx"] == _MISSED[0]) & (stored["end_idx"] == _MISSED[1])].iloc[0]
    assert float(row["depth_pA"]) == pytest.approx(float(preview.row["depth_pA"]), rel=1e-9)


def test_a_poisoned_flank_does_not_blank_the_readout_when_the_table_supplies_the_baseline(
        win, tmp_path):
    signal = _signal(_DETECTED + [_MISSED])
    signal[19_900:19_950] = np.nan                           # beside the cursors, not under them
    doc_id = win._session.add_document(_doc(tmp_path, signal, name="poisoned.dat"))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    mp = _bracket(win, doc_id)
    assert mp.current_measurement() is not None and mp.refusal_note() == ""
    assert mp.event_preview() is not None and mp.event_preview().table == "Manual"


# ------------------------------------------------------------------------------------------------
# an owner that fails is heard, and the close prompt knows a Manual table is work
# ------------------------------------------------------------------------------------------------
def test_a_failing_provider_says_so_on_the_preview_line_and_the_readout_survives(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    mp = win._analysis_panel.measure_panel

    def broken(*_a, **_k):
        raise RuntimeError("boom")

    mp.set_basis_provider(broken)
    _bracket(win, doc_id)
    text = mp.event_preview_text()
    assert text.startswith("No event can be stored here: the preview failed (RuntimeError: boom)")
    assert mp.current_measurement() is not None             # measured on its own, as before


def test_the_hand_made_counts_name_every_table_that_has_some(win, tmp_path):
    doc_id = _open(win, tmp_path)                           # a Schmitt table is shown
    panel = win._analysis_panel
    assert panel.hand_made_summary(doc_id, 0) == "No hand-made events yet"
    assert panel.add_manual_event(doc_id, 0, *_MISSED).ok                    # into Schmitt
    selector = panel._result_selector
    selector.setCurrentIndex(selector.findData(MANUAL_MODULE_ID))
    assert panel.add_manual_event(doc_id, 0, 40_000, 40_300).ok              # into Manual
    assert panel.hand_made_summary(doc_id, 0) == "Hand-made here: Schmitt 1, Manual 1"


def test_closing_a_recording_with_only_a_manual_table_asks_first(win, tmp_path):
    doc_id = _open(win, tmp_path, detect=False)
    assert win._analysis_panel.add_manual_event(doc_id, 0, *_MISSED).ok
    sentence, _segment = win._describe_close_stake([doc_id])
    assert sentence and "1 hand-added" in sentence
