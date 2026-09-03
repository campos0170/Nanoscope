"""The two seams the Measure panel had built and could not reach, driven end to end.

W-MEASUREBASIS. ``MeasurePanel.set_conditioning`` was a door with no caller: the
"measure on" combo was on screen, the memo and the refusals worked, and choosing
*Processed signal* measured nothing new because ``self._conditioning`` was ``None`` for
the life of the process. The panel is not allowed to ask a sibling panel for the
Preprocess form, but it does not have to: the chain a detection on this very channel ran
under is in ``AnalysisResult.params``, ``Session`` owns the result store, and
``analysis.conditioning.conditioned_trace`` is a pure function of ``(raw, fs, params)``.
So the panel recovers the chain itself. These tests assert the USER-VISIBLE EFFECT — a
selectable combo item, a sentence naming the chain, and a number that moves — never that
some method was called.

USABLE-1b. The processed-trace export names its CSV column ``"current_" + units``. On a
command-voltage channel that wrote ``current_mV`` under a sidecar saying
``"export": "processed_trace"``: every field true, the file a lie. It is refused now, and
the neighbouring states are pinned beside it — a normalized chain (labelled
dimensionless), an undeclared-units channel (``current``, claiming nothing), and the
calibration stamp that makes the sidecar a recipe instead of a description.

WHAT FAILS BEFORE THE WIRING. Every assertion here is on the effect, so:
``processed_item_enabled`` was False forever (no chain could reach the panel);
``processed_export_gate`` never returned "" and never mentioned a channel kind;
``export_processed_trace`` wrote ``time_s,current_mV`` instead of refusing; and the
sidecar's ``cal_gain``/``cal_offset`` were ``null`` on every export the door ever wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis.conditioning import BASIS_PROCESSED, BASIS_RAW, conditioned_trace
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.measure_panel import MeasurePanel
from nanoscope.gui.trace_view import TraceView

FS = 50_000.0
CHAIN = {"filter_cutoff_hz": 5000.0, "filter_kind": "bessel", "filter_order": 8}
REAL_ABF = (
    "/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
    "0004_process _ -1000 mV_ 385-400 s time trace.abf"
)


# ================================================================================================
# builders
# ================================================================================================
def _trace(n: int = 120_000, seed: int = 11) -> np.ndarray:
    """An open pore at -600 pA with real noise and six blockades.

    Noisy on purpose: the whole point of measuring on the conditioned trace is that a
    low-pass moves the SPREAD statistics, and a clean trace would let a broken wiring
    pass by returning almost the same numbers.
    """
    rng = np.random.default_rng(seed)
    x = -600.0 + rng.normal(0.0, 12.0, n)
    for start in (10_000, 30_000, 50_000, 70_000, 90_000, 110_000):
        x[start:start + 400] -= 90.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(tmp_path, data: np.ndarray, *, kind=ChannelKind.CURRENT, units="pA",
         name="pore.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="IN15", raw_units=units, kind=kind)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=data.size,
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(data,))


def _stored_detection(doc_id: str, channel: int, params: dict) -> AnalysisResult:
    """A detection result carrying ``params`` — the shape ``DetectSchmitt.run`` returns.

    Only ``params`` matters here: it is the Preprocess/Detection form as it stood when
    that run was launched, and it is the one thing the panel reads back out.
    """
    return AnalysisResult(
        module_id="detect.schmitt", module_version="test", doc_id=doc_id,
        channel=int(channel), params=params, calibration_fingerprint=(1.0, 0.0))


@pytest.fixture
def panel(qapp, qtbot, tmp_path):
    """A real MeasurePanel on a real TraceView showing one synthetic current channel."""
    session = Session()
    data = _trace()
    doc_id = session.add_document(_doc(tmp_path, data))
    session.add_layer(doc_id, 0)
    tv = TraceView(session)
    qtbot.addWidget(tv)
    tv.resize(1000, 400)
    tv.show()
    qtbot.waitExposed(tv)
    mp = MeasurePanel(session, tv)
    qtbot.addWidget(mp)
    mp.resize(520, 900)
    mp.show()
    qtbot.waitExposed(mp)
    mp.set_selection(doc_id, 0)
    return mp, session, doc_id, data


def _processed_item_enabled(mp: MeasurePanel) -> bool:
    """Is *Processed signal* selectable? — what a user sees when the combo drops open."""
    model = mp._basis_combo.model()
    return model.item(mp._basis_combo.findData(BASIS_PROCESSED)).isEnabled()


def _measure(mp: MeasurePanel, doc_id: str, start: int, stop: int) -> dict:
    mp.set_active(True)
    assert mp._place_cursors(doc_id, 0, start, stop) is not None
    return mp.exact_measurement()


# ================================================================================================
# W-MEASUREBASIS — the basis combo stops being a control that toggles nothing
# ================================================================================================
def test_a_stored_detections_chain_makes_processed_selectable_and_says_whose_it_is(panel):
    """Before: greyed forever. After: selectable, with the chain and its SOURCE named."""
    mp, session, doc_id, _data = panel

    assert not _processed_item_enabled(mp), "no chain anywhere yet"
    assert "chain IS available" not in mp.basis_note()

    session.add_result(_stored_detection(doc_id, 0, dict(CHAIN)))
    mp.set_detection_present(True)          # the push the owner already makes

    assert _processed_item_enabled(mp)
    note = mp.basis_note()
    assert "low-pass bessel 5000 Hz (order 8)" in note
    # ...and it does NOT claim the Preprocess form, because it is not the form.
    assert "stored detection" in note
    assert mp.conditioning_params()["filter_cutoff_hz"] == 5000.0


def test_choosing_processed_measures_the_conditioned_array_not_the_raw_one(panel):
    """The number on screen moves, and moves to the value the pure kernel gives."""
    mp, session, doc_id, data = panel
    session.add_result(_stored_detection(doc_id, 0, dict(CHAIN)))
    mp.set_detection_present(True)

    raw = _measure(mp, doc_id, 20_000, 60_000)
    mp.set_measurement_basis(BASIS_PROCESSED)
    proc = mp.exact_measurement()

    assert raw is not None and proc is not None
    assert proc["n_samples"] == raw["n_samples"] == 40_000
    # A 5 kHz low-pass on 50 kHz noise cuts the spread; nothing subtle about it.
    assert proc["std"] < raw["std"] * 0.9
    # And it is EXACTLY the conditioned array, not "something filtered".
    expect = conditioned_trace(np.asarray(data, dtype=np.float64), FS, CHAIN)[20_000:60_000]
    assert proc["std"] == pytest.approx(float(np.std(expect)), rel=1e-6)
    assert proc["mean"] == pytest.approx(float(np.mean(expect)), rel=1e-6)
    assert mp.readout_value("std") != ""


def test_a_live_push_beats_the_stored_run_and_a_row_records_which_basis_it_was(panel):
    """``set_conditioning`` still wins when it carries a chain — it is the live form."""
    mp, session, doc_id, _data = panel
    session.add_result(_stored_detection(doc_id, 0, dict(CHAIN)))
    mp.set_detection_present(True)
    assert mp.conditioning_params()["filter_cutoff_hz"] == 5000.0

    mp.set_conditioning({"filter_cutoff_hz": 1000.0, "filter_kind": "bessel",
                         "filter_order": 8})
    assert mp.conditioning_params()["filter_cutoff_hz"] == 1000.0
    assert "stored detection" not in mp.basis_note()

    _measure(mp, doc_id, 20_000, 60_000)
    mp.set_measurement_basis(BASIS_PROCESSED)
    assert mp.add_current() is True
    row = mp.measurements_df.iloc[-1]
    assert row["signal_basis"] == BASIS_PROCESSED

    mp.set_measurement_basis(BASIS_RAW)
    assert mp.add_current() is True
    assert mp.measurements_df.iloc[-1]["signal_basis"] == BASIS_RAW


def test_dropping_the_detection_takes_the_processed_basis_away_again(panel):
    """The fallback is live in both directions; it does not latch a chain forever."""
    mp, session, doc_id, _data = panel
    session.add_result(_stored_detection(doc_id, 0, dict(CHAIN)))
    mp.set_detection_present(True)
    assert _processed_item_enabled(mp)

    mp.set_measurement_basis(BASIS_PROCESSED)
    session.results.drop(doc_id, 0, "detect.schmitt")
    mp.set_detection_present(False)
    assert not _processed_item_enabled(mp)
    # The choice is LEFT STANDING and refused out loud; it is not silently reset to raw.
    assert mp.measurement_basis() == BASIS_PROCESSED
    assert "no stored detection" in mp.basis_note()


# ================================================================================================
# USABLE-1b — the export door refuses a channel it cannot label honestly
# ================================================================================================
def test_the_export_refuses_a_command_voltage_channel_instead_of_heading_it_current_mV(
        qapp, qtbot, tmp_path):
    """The whole defect, in one file that is never written."""
    session = Session()
    doc_id = session.add_document(
        _doc(tmp_path, _trace(), kind=ChannelKind.VOLTAGE, units="mV", name="cmd.dat"))
    session.add_layer(doc_id, 0)
    tv = TraceView(session)
    qtbot.addWidget(tv)
    mp = MeasurePanel(session, tv)
    qtbot.addWidget(mp)
    mp.set_selection(doc_id, 0)
    mp.set_conditioning(dict(CHAIN))

    gate = mp.processed_export_gate()
    assert gate, "a voltage channel must not be exportable as a processed CURRENT trace"
    assert "current_mV" in gate, "the refusal quotes the false header it would have written"
    assert "IN15" in gate and "VOLTAGE" in gate
    assert "Calibration tab" in gate          # the sentence names a way out

    out = mp.export_processed_trace(str(tmp_path / "volt.csv"))
    assert out is None
    assert not (tmp_path / "volt.csv").exists()
    assert not (tmp_path / "volt.meta.json").exists()
    assert "Nothing exported" in mp._status.text()


def test_the_calibration_override_decides_the_kind_both_ways(qapp, qtbot, tmp_path):
    """``effective_kind``, not the header — the resolver every other surface asks.

    A file that declared its analyte channel a voltage is exportable once the user has
    corrected it on the Calibration tab, and a real current channel the user has just
    declared a command voltage stops being exportable at the same instant.
    """
    session = Session()
    doc_id = session.add_document(
        _doc(tmp_path, _trace(), kind=ChannelKind.VOLTAGE, units="mV", name="cmd.dat"))
    session.add_layer(doc_id, 0)
    tv = TraceView(session)
    qtbot.addWidget(tv)
    mp = MeasurePanel(session, tv)
    qtbot.addWidget(mp)
    mp.set_selection(doc_id, 0)
    mp.set_conditioning(dict(CHAIN))
    assert mp.processed_export_gate()

    session.set_calibration(
        doc_id, 0, ChannelCalibration(kind=ChannelKind.CURRENT, target_units="pA"),
        write_sidecar=False)
    mp.set_selection(doc_id, 0)
    assert mp.processed_export_gate() == ""
    out = mp.export_processed_trace(str(tmp_path / "ok.csv"))
    assert out is not None
    assert open(out["data"]).readline().strip() == "time_s,current_pA"

    session.set_calibration(
        doc_id, 0, ChannelCalibration(kind=ChannelKind.VOLTAGE, target_units="mV"),
        write_sidecar=False)
    assert "current_mV" in mp.processed_export_gate()


def test_the_neighbouring_states_are_labelled_rather_than_refused(panel, tmp_path):
    """Normalized, undeclared units, and the calibration the chain ran under.

    None of these is a refusal: they are all a CURRENT channel, so the column may say
    current — it just has to say the right thing after it.
    """
    mp, session, doc_id, _data = panel

    # (1) normalize takes the picoamps with it, and the header says so.
    mp.set_conditioning({**CHAIN, "normalize_enabled": True, "normalize_mode": "zscore"})
    assert mp.processed_export_gate() == ""
    out = mp.export_processed_trace(str(tmp_path / "norm.csv"))
    assert open(out["data"]).readline().strip() == \
        "time_s,current_pA_normalized_(dimensionless)"
    assert json.load(open(out["meta"]))["units"] == "pA normalized (dimensionless)"

    # (2) THE SIDECAR IS A RECIPE. The chain runs on the CALIBRATED channel, so the two
    # numbers that alter the array have to be in the file; they were null on every export
    # this door had written.
    session.set_calibration(doc_id, 0,
                            ChannelCalibration(gain=2.5, offset=-3.0, target_units="pA"),
                            write_sidecar=False)
    mp.set_conditioning(dict(CHAIN))
    out = mp.export_processed_trace(str(tmp_path / "cal.csv"))
    meta = json.load(open(out["meta"]))
    assert meta["cal_gain"] == 2.5 and meta["cal_offset"] == -3.0
    assert meta["units"] == "pA"
    assert open(out["data"]).readline().strip() == "time_s,current_pA"


def test_an_undeclared_units_current_channel_claims_no_unit(qapp, qtbot, tmp_path):
    """No units on disk and none set: the column is ``current`` and states nothing."""
    session = Session()
    doc_id = session.add_document(
        _doc(tmp_path, _trace(), kind=ChannelKind.CURRENT, units="", name="bare.dat"))
    session.add_layer(doc_id, 0)
    tv = TraceView(session)
    qtbot.addWidget(tv)
    mp = MeasurePanel(session, tv)
    qtbot.addWidget(mp)
    mp.set_selection(doc_id, 0)
    mp.set_conditioning(dict(CHAIN))
    assert mp.processed_export_gate() == ""
    out = mp.export_processed_trace(str(tmp_path / "bare.csv"))
    assert open(out["data"]).readline().strip() == "time_s,current"
    assert json.load(open(out["meta"]))["units"] == ""
    assert "no declared units" in mp._status.text()


# ================================================================================================
# the owner's own recording, in the real window
# ================================================================================================
@pytest.mark.skipif(not Path(REAL_ABF).exists(),
                    reason="the owner's -1000 mV apo recording is not on this machine")
def test_the_real_window_offers_the_processed_basis_and_the_numbers_move(qapp, qtbot):
    """1512x900, his 15 s apo file, the combo operated the way a user operates it.

    The detection result is installed directly rather than run (the run is 60 s and this
    test is about the PANEL, not the detector); its ``params`` is the same dict the run
    stores.
    """
    from nanoscope.gui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1512, 900)
    win.show()
    qtbot.waitExposed(win)
    win.open_paths([Path(REAL_ABF)])
    qtbot.waitUntil(lambda: len(win._session.layers()) > 0, timeout=30_000)
    qtbot.wait(100)

    ap = win._analysis_panel
    win._inspector_tabs.setCurrentWidget(ap)
    qtbot.wait(50)
    mp = ap.measure_panel
    doc_id, channel = mp._target
    assert win._session.effective_kind(doc_id, channel) is ChannelKind.CURRENT

    assert not _processed_item_enabled(mp)
    win._session.add_result(_stored_detection(doc_id, channel, dict(CHAIN)))
    mp.set_detection_present(True)
    assert _processed_item_enabled(mp), "the combo item a user has to be able to pick"

    mp.set_active(True)
    assert mp._place_cursors(doc_id, channel, 100_000, 140_000) is not None
    raw = mp.exact_measurement()
    mp.set_measurement_basis(BASIS_PROCESSED)
    proc = mp.exact_measurement()
    assert raw["n_samples"] == proc["n_samples"] == 40_000
    assert proc["std"] < raw["std"]
    assert proc["ptp"] < raw["ptp"] * 0.98
    assert "low-pass bessel 5000 Hz" in mp.basis_note()
    assert mp.processed_export_gate() == ""
