"""The Sample rate / time unit dialog and the seams around it: File Info and the status bar
show the declaration as an override, a real MainWindow saves and restores it through the
project, and a detection made at the old rate is marked stale rather than re-scaled."""
from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest

from nanoscope.analysis.measure import append_measurement, empty_measurements_df
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.project import load_project, save_project
from nanoscope.core.segment import persistent_key
from nanoscope.core.sample_rate import SampleRateDeclaration
from nanoscope.core.sample_rate_state import WORKSPACE_KEY
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.file_info_panel import FileInfoPanel
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.sample_rate_dialog import SampleRateDialog

FS = 100_000.0
N = 40_000


def _trace(seed: int = 0) -> np.ndarray:
    """A quiet baseline with three unmistakable blockades — enough for detection."""
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in [(4_000, 4_300), (12_000, 12_250), (25_000, 25_400)]:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(path: Path, *, rate: float | None = 100.0,
         prov: Provenance = Provenance.MEASURED, refused: float | None = None,
         fmt: str = "Text trace") -> TraceDocument:
    """A 100 kHz trace whose time column was in milliseconds and read as seconds: 100 Hz."""
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=path, format_name=fmt, n_samples=arr.size,
                         sample_rate_hz=rate, rate_provenance=prov, channels=(info,),
                         t0_s=0.05, refused_rate_hz=refused)
    return TraceDocument(meta=meta, channels=(arr,))


def _detect(window, qtbot, doc):
    key = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(key, 0)
    panel = window._analysis_panel
    panel.detection_panel.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    qtbot.wait(20)
    assert panel.current_detection() is not None
    return key, panel


@pytest.fixture
def dialog_rig(qapp, qtbot, tmp_path):
    path = tmp_path / "recording.csv"
    path.write_bytes(b"unchanged source fixture")
    session = Session()
    doc = _doc(path)
    key = session.add_document(doc)
    dialog = SampleRateDialog(session, key)
    qtbot.addWidget(dialog)
    return dialog, session, doc, key, path


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


# -- the dialog ----------------------------------------------------------------------------------

def test_the_dialog_declares_a_unit_then_a_rate_then_returns_to_the_file(dialog_rig):
    dialog, session, doc, key, path = dialog_rig
    assert "100 Hz · measured" in dialog.header.text()
    assert "read as seconds" in dialog.header.text()
    assert dialog.mode.currentData() == "file"
    assert not dialog.unit.isEnabled() and not dialog.rate.isEnabled()

    dialog.mode.setCurrentIndex(dialog.mode.findData("time_unit"))
    dialog.unit.setCurrentIndex(dialog.unit.findData("ms"))
    dialog.note.setPlainText("the exporter writes milliseconds")
    assert "100,000 Hz" in dialog.preview.text() and "5e-05 s" in dialog.preview.text()
    dialog.apply()
    assert dialog.error.text().startswith("Declared: 100,000 Hz.")
    meta = session.document(key).meta
    assert meta.sample_rate_hz == pytest.approx(FS)
    assert meta.rate_provenance is Provenance.USER
    assert meta.t0_s == pytest.approx(5e-5)
    assert session.sample_rate_declaration(key).note == "the exporter writes milliseconds"
    assert "100 Hz · measured" in dialog.header.text(), "the file's reading stays on screen"

    reopened = SampleRateDialog(session, key)
    assert reopened.mode.currentData() == "time_unit"
    assert reopened.unit.currentData() == "ms"
    assert reopened.note.toPlainText() == "the exporter writes milliseconds"

    dialog.mode.setCurrentIndex(dialog.mode.findData("sample_rate"))
    dialog.rate.setText("12345")
    dialog.apply()
    assert session.document(key).meta.sample_rate_hz == 12_345.0
    assert session.document(key).meta.t0_s == pytest.approx(0.05), "a rate leaves t0 as read"

    dialog.mode.setCurrentIndex(dialog.mode.findData("file"))
    dialog.apply()
    assert "file's own time base" in dialog.error.text()
    assert session.document(key).meta.sample_rate_hz == 100.0
    assert session.document(key).meta.rate_provenance is Provenance.MEASURED
    assert session.sample_rate_declaration(key) is None
    assert path.read_bytes() == b"unchanged source fixture"
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


@pytest.mark.parametrize("rate", ["", " ", "0", "-1", "nan", "inf", "1e999", "not a number",
                                  "10 kHz", "1_000", "١٠٠٠"])
def test_an_invalid_rate_is_refused_without_changing_the_time_base(dialog_rig, rate):
    dialog, session, doc, key, path = dialog_rig
    dialog.mode.setCurrentIndex(dialog.mode.findData("sample_rate"))
    dialog.rate.setText(rate)
    dialog.apply()
    assert dialog.error.text() == (
        "Not applied: the sample rate must be a finite positive number in Hz")
    assert session.sample_rate_declaration(key) is None
    assert session.document(key).meta.sample_rate_hz == 100.0
    assert path.read_bytes() == b"unchanged source fixture"


def test_the_dialog_offers_a_unit_for_a_refused_column_but_not_for_a_header_declared_rate(
        qapp, qtbot, tmp_path):
    """The microsecond column the floor refused: 'refused' in the header, the unit mode
    offered (its interval was kept) and the preview showing the recovered clock. An ABF's
    header-declared rate: no column to re-read, so only a rate can be declared."""
    session = Session()
    refused = session.add_document(_doc(tmp_path / "us.csv", rate=None, prov=Provenance.UNKNOWN,
                                        refused=0.1))
    dialog = SampleRateDialog(session, refused)
    qtbot.addWidget(dialog)
    assert dialog.header.text().startswith("refused — the time column steps by 10 per sample")
    assert dialog.mode.model().item(dialog.mode.findData("time_unit")).isEnabled()
    dialog.mode.setCurrentIndex(dialog.mode.findData("time_unit"))
    dialog.unit.setCurrentIndex(dialog.unit.findData("us"))
    assert "100,000 Hz" in dialog.preview.text()
    dialog.apply()
    assert session.document(refused).meta.sample_rate_hz == pytest.approx(FS)
    assert session.document(refused).meta.t0_s == pytest.approx(0.05e-6)

    h5 = session.add_document(_doc(tmp_path / "us.h5", rate=None, prov=Provenance.UNKNOWN,
                                   refused=0.1, fmt="HDF5 flat"))
    assert SampleRateDialog(session, h5).header.text().startswith("refused — /time steps by 10")
    info = FileInfoPanel(session)
    info.set_document(h5)
    assert info._fields["t₀"].text() == "0.05 as written (unit undeclared)", (
        "a refused time base's first value is never labelled seconds")
    session.set_sample_rate_declaration(h5, SampleRateDeclaration("time_unit", time_unit="us"))
    assert info._fields["t₀"].text() == "5e-08 s"

    abf = session.add_document(_doc(tmp_path / "a.abf", rate=FS, prov=Provenance.FILE_META))
    other = SampleRateDialog(session, abf)
    qtbot.addWidget(other)
    assert "declared in the file's header" in other.header.text()
    assert not other.mode.model().item(other.mode.findData("time_unit")).isEnabled()
    other.mode.setCurrentIndex(other.mode.findData("time_unit"))
    assert other.preview.text().startswith("Cannot apply:")
    other.apply()
    assert other.error.text().startswith("Not applied: this recording's rate is declared")
    assert session.document(abf).meta.sample_rate_hz == FS


def test_a_segment_s_dialog_declines_and_file_info_says_it_follows(qapp, qtbot, tmp_path):
    session = Session()
    key = session.add_document(_doc(tmp_path / "pore.csv"))
    session.set_sample_rate_declaration(key, SampleRateDeclaration("time_unit", time_unit="ms"))
    seg = session.create_segment(key, 1000, 3000, name="cut")
    panel = FileInfoPanel(session)
    panel.set_document(seg)
    assert panel._fields["Sample rate"].text() == (
        "100,000 Hz  ·  user  (follows the recording it was cut from)")
    dialog = SampleRateDialog(session, seg)
    qtbot.addWidget(dialog)
    assert dialog.header.text() == (
        "100,000 Hz · user · t₀ = 0 s (the declaration made on the recording this was cut "
        "from) — a segment, on its parent's clock")
    assert not dialog.mode.isEnabled()
    assert dialog.error.text().startswith("This is a segment")
    dialog.apply()
    assert dialog.error.text().startswith("Not applied: a segment follows")
    assert session.document(seg).meta.sample_rate_hz == pytest.approx(FS)


def test_a_rateless_recording_offers_only_a_rate(qapp, qtbot, tmp_path):
    session = Session()
    key = session.add_document(_doc(tmp_path / "legacy.tdms", rate=None, prov=Provenance.UNKNOWN))
    dialog = SampleRateDialog(session, key)
    qtbot.addWidget(dialog)
    assert dialog.header.text().startswith("unknown")
    assert not dialog.mode.model().item(dialog.mode.findData("time_unit")).isEnabled()
    dialog.mode.setCurrentIndex(dialog.mode.findData("sample_rate"))
    dialog.rate.setText("10000")
    dialog.apply()
    assert session.document(key).meta.sample_rate_hz == 10_000.0
    assert session.x_axis(key).unit == "s"


# -- the surfaces that print the rate ------------------------------------------------------------

def test_file_info_and_the_status_bar_show_the_override_with_the_file_reading(window, tmp_path):
    doc = _doc(tmp_path / "pore.csv")
    key = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(key, 0)
    panel = FileInfoPanel(window._session)
    panel.set_document(key)
    assert panel._fields["Sample rate"].text() == "100 Hz  ·  measured"
    assert panel._edit_rate.isEnabled()
    window._dirty = False

    window._session.set_sample_rate_declaration(key, SampleRateDeclaration("time_unit", time_unit="ms"))
    assert window._dirty, "a declaration is project state"
    assert panel._fields["Sample rate"].text() == "100,000 Hz  ·  user  (file: 100 Hz · measured)"
    assert panel._fields["Duration"].text() == f"{N / FS:.6g} s"
    assert window._rate_label.text() == "100,000 Hz · user"
    assert window._session.x_axis(key).values[1] == pytest.approx(5e-5 + 1 / FS)

    window._session.set_sample_rate_declaration(key, None)
    assert panel._fields["Sample rate"].text() == "100 Hz  ·  measured"
    assert window._rate_label.text() == "100 Hz · measured"
    window._dirty = False


# -- the project -------------------------------------------------------------------------------

def test_real_main_window_saves_and_restores_the_declaration(qapp, qtbot, tmp_path):
    source = MainWindow()
    qtbot.addWidget(source)
    path = tmp_path / "recording.csv"
    path.write_bytes(b"unchanged source fixture")
    doc = _doc(path)
    key = source._session.add_document(doc)
    source._create_default_layers(doc)
    source._on_selection(key, 0)
    source._dirty = False
    declaration = SampleRateDeclaration("time_unit", time_unit="ms", note="Confirmed")
    source._session.set_sample_rate_declaration(key, declaration)
    assert source._dirty
    state = source._gather_project_state()
    assert state.workspace.extra[WORKSPACE_KEY] == [
        {"doc_id": str(path.resolve()), "declaration": declaration.to_dict()}]
    saved = save_project(tmp_path / "workflow.nanoscope", state)

    destination = MainWindow()
    qtbot.addWidget(destination)
    other_doc = _doc(path)
    other_key = destination._session.add_document(other_doc)
    destination._restore_state = load_project(saved)
    destination._apply_restore_for_doc(other_key)
    assert destination._session.sample_rate_declaration(other_key) == declaration
    assert destination._session.document(other_key).meta.sample_rate_hz == pytest.approx(FS)
    assert destination._session.document(other_key).meta.rate_provenance is Provenance.USER
    assert other_doc.meta.sample_rate_hz == 100.0, "the document as read is untouched"
    assert path.read_bytes() == b"unchanged source fixture"
    assert not (tmp_path / "recording.csv.npcal.json").exists()
    source._dirty = False
    destination._dirty = False
    destination._restore_state = None


# -- the trace view ----------------------------------------------------------------------------

def test_the_trace_view_frames_the_new_clock_and_forgets_the_old_one(window, qtbot, tmp_path):
    """Measured before the fix: zoomed to [119, 124] s on the 100 Hz-as-read file, declaring
    milliseconds rebuilt the view at the OLD range over data that now end at 0.4 s — a blank
    pane under a seconds axis, and a Zoom Back trail leading to 400 s."""
    doc = _doc(tmp_path / "pore.csv")
    key = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(key, 0)
    view = window._trace_view
    anchor = view.anchor_plot()
    assert anchor is not None
    anchor.vb.setXRange(119.0, 124.0, padding=0)
    qtbot.wait(350)                                   # past the trail's settle debounce
    assert 118.0 < anchor.vb.viewRange()[0][0] < 120.0
    assert any(118.0 < lo < 120.0 for lo, _hi in view._view_hist), "the zoom is on the trail"
    # The Measure region sits on one blockade (samples 11,995–12,255) with its edges dragged
    # past the half-sample mark, as a hand-placed region's are; it must stay there.
    view.set_measure_active(key, 0, True)
    view._measure_region.setRegion((0.05 + 11_994.6 / 100.0, 0.05 + 12_255.4 / 100.0))
    qtbot.wait(20)
    assert view.measure_bounds() == (11_995, 12_255)

    window._session.set_sample_rate_declaration(key, SampleRateDeclaration("time_unit", time_unit="ms"))
    anchor = view.anchor_plot()
    lo, hi = anchor.vb.viewRange()[0]
    extent = N / FS
    assert lo >= -0.1 * extent and hi <= 1.1 * extent, (lo, hi)
    assert len(view._view_hist) == 1 and -0.1 * extent <= view._view_hist[0][0] <= extent, (
        "the Back/Forward trail was in the old unit")
    assert view.measure_bounds() == (11_995, 12_255), "carried by sample index, never re-seeded"

    # A declaration on a recording that is NOT drawn rebuilds nothing.
    other = _doc(tmp_path / "other.csv")
    other_key = window._session.add_document(other)
    calls = []
    view.rebuild = lambda _c=calls: _c.append(1)
    window._session.set_sample_rate_declaration(other_key, SampleRateDeclaration("time_unit", time_unit="ms"))
    assert calls == []
    window._dirty = False


# -- staleness ---------------------------------------------------------------------------------

def test_a_stored_cutoff_above_the_new_nyquist_refuses_re_run_before_the_click(window, qtbot,
                                                                                 tmp_path):
    """Declaring 1 kHz under a detection made at 100 kHz puts every stored cutoff above the
    new Nyquist (500 Hz): the banner must say so and gate the button, not launch a run the
    kernel refuses after the click has dismissed the banner."""
    doc = _doc(tmp_path / "pore.csv", rate=FS, prov=Provenance.FILE_META)
    key = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(key, 0)
    panel = window._analysis_panel
    panel.detection_panel.filter_enabled.setChecked(True)        # a 10 kHz low-pass, stored
    panel.detection_panel.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    qtbot.wait(20)
    res = panel.current_detection()
    assert res is not None and res.params.get("filter_cutoff_hz") == 10_000.0
    window._session.set_sample_rate_declaration(
        key, SampleRateDeclaration("sample_rate", sample_rate_hz=1_000.0))
    assert panel._stale.kind == "sample_rate"
    why = panel._stale.action_refusal()
    assert "Nyquist" in why and "500 Hz" in why and "Detect" in why, why
    # ...and at a declared rate the stored cutoff IS below Nyquist, the button comes back.
    window._session.set_sample_rate_declaration(
        key, SampleRateDeclaration("sample_rate", sample_rate_hz=50_000.0))
    assert panel._stale.kind == "sample_rate" and panel._stale.action_refusal() == ""
    # An explicit notch list with no line below the new Nyquist is the same refusal.
    window._session.set_sample_rate_declaration(
        key, SampleRateDeclaration("sample_rate", sample_rate_hz=100.0))
    why = panel._cutoff_replay_why({"notch_enabled": True, "notch_freqs": [60.0, 120.0]})
    assert why is not None and "notch lines" in why and "50 Hz" in why and "Detect" in why
    assert panel._cutoff_replay_why({"notch_enabled": True, "notch_freqs": [40.0, 120.0]}) is None
    window._dirty = False


def test_a_rate_change_keeps_the_calibration_badges_it_arrives_on_top_of(window, qtbot, tmp_path):
    key, panel = _detect(window, qtbot, _doc(tmp_path / "pore.csv", rate=FS,
                                             prov=Provenance.FILE_META))
    window._session.set_calibration(key, 0, ChannelCalibration(gain=1.1), write_sidecar=False)
    assert panel._stale.kind == "calibration"
    assert "depth_pA" in panel._table.stale_columns
    window._session.set_sample_rate_declaration(
        key, SampleRateDeclaration("sample_rate", sample_rate_hz=50_000.0))
    assert panel._stale.kind == "sample_rate"
    assert "calibration changed since this detection ran as well" in panel._stale.message()
    assert panel._stale._verdict.endswith("rate and calibration changed")
    cols = panel._table.stale_columns
    assert "dwell_s" in cols and "depth_pA" in cols and "baseline_pA" in cols
    window._dirty = False


def test_a_manual_measurement_made_at_the_old_rate_wears_the_mark(window, qtbot, tmp_path):
    """A saved row stores its span in samples beside its seconds; after the declaration the
    two disagree, and the row says so — never rescaled."""
    doc = _doc(tmp_path / "pore.csv")
    key = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(key, 0)
    measure = window._analysis_panel.measure_panel
    row = {"recording": doc.display_name, "doc_key": persistent_key(doc), "channel": "current",
           "channel_index": 0, "start_idx": 1000, "end_idx": 3000, "t_start_s": 10.05,
           "dwell_s": 20.0, "n_samples": 2000, "units": "pA", "cal_gain": 1.0,
           "cal_offset": 0.0, "t0_s": 0.05}
    measure.set_measurements_df(append_measurement(empty_measurements_df(), row))
    measure._refresh_stale_rows()
    assert measure._model._row_states == {}, "measured at the live rate: current"
    window._session.set_sample_rate_declaration(key, SampleRateDeclaration("time_unit", time_unit="ms"))
    assert measure._model._row_states == {0: "rate_stale"}
    assert measure.measurements_df.loc[0, "dwell_s"] == 20.0, "never rescaled"
    assert "time base that has since been re-declared" in measure._model._row_tip(0)
    window._session.set_sample_rate_declaration(key, None)
    assert measure._model._row_states == {}
    # ...and a row found rate-stale keeps a mark that says WHAT moved after its recording
    # closes, rather than lapsing into the calibration sentence.
    window._session.set_sample_rate_declaration(key, SampleRateDeclaration("time_unit", time_unit="ms"))
    window._session.close_document(key)
    measure._refresh_stale_rows()
    assert measure._model._row_states == {0: "rate_lapsed"}
    tip = measure._model._row_tip(0)
    assert "time base that had already been re-declared" in tip and "calibration" not in tip
    window._dirty = False


def test_a_detection_made_at_the_old_rate_is_marked_stale_not_rescaled(window, qtbot, tmp_path):
    key, panel = _detect(window, qtbot, _doc(tmp_path / "pore.csv", rate=FS,
                                             prov=Provenance.FILE_META))
    res = panel.current_detection()
    assert res is not None and len(res.payload.df) == 3
    before = res.payload.df.copy(deep=True)
    assert panel._stale.isHidden()

    window._session.set_sample_rate_declaration(
        key, SampleRateDeclaration("sample_rate", sample_rate_hz=50_000.0))
    assert not panel._stale.isHidden(), "the banner lives in the Results dock, not under the panel"
    assert panel._stale.kind == "sample_rate"
    said = panel._stale.message()
    assert "100,000 Hz → 50,000 Hz" in said and "old rate" in said, said
    assert panel._stale.action_text() == "Re-run"
    assert panel._stale._verdict.endswith("Times stale — sample rate changed")
    assert "dwell_s" in panel._table.stale_columns and "t_start_s" in panel._table.stale_columns
    assert "depth_pA" not in panel._table.stale_columns, "the amplitude columns did not move"
    after = panel.current_detection().payload.df
    assert after.equals(before), "never silently re-scaled"
    assert panel.detection_panel.sample_rate() == pytest.approx(50_000.0), (
        "the form's Nyquist-bounded controls follow the declared rate")

    window._session.set_sample_rate_declaration(key, None)
    assert panel._stale.isHidden()
    assert panel._table.stale_columns == ()
    window._dirty = False
