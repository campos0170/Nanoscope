"""FE-02 + FE-04 — the detection Summary says which rows it counted and what unit it is in.

**FE-02.** ``AnalysisPanel._fill_summary`` masked exactly one number by ``accepted``: the
rate. The four medians below it were taken over the WHOLE frame, and
``_refresh_summary_from_table`` re-ran the same unmasked arithmetic on every curation
edit, so the strings did not merely go stale — they were recomputed and landed
byte-identical. Rejecting the 30 deepest of 115 events moved "n accepted" 115 → 85 and
"rate (Hz, accepted)" 7.67 → 5.669 and left "median depth (pA) 90.33" (accepted rows:
82.40) directly underneath, with nothing on screen saying the neighbouring rows count
different sets. The Plots window opened on the same detection filters to accepted events
BY DEFAULT, so the two surfaces disagreed about the same column of the same table.

**FE-04.** The two amplitude rows captioned themselves "(pA)" unconditionally. On the
owner's ``2_3 electroporation-005_Cut_1.h5`` — ``raw_units=''``, a loader note reading
"Magnitudes suggest nA", a status chip reading "Current gain uncalibrated" — the Summary
read "median depth (pA) 0.07274" while the PSD window on the same channel in the same
session read "0.004204 (units unknown)"; and after the user did what every surface tells
him to do and declared the channel nA in Calibration, the PSD followed and the Summary
still said "(pA)". The units-honesty guard the Calibration panel, the I-V window and the
PSD window already carry is now on this panel too, in the same words.

The third rule these tests pin is the one the measurement table learned earlier: a gain
edit must not RE-CAPTION numbers that were measured under the old gain. A units-only
edit (``target_units`` is a label; it does not touch the array) does re-caption at once —
that is the two surfaces agreeing — but a gain edit leaves the caption alone and raises
the staleness banner instead.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6 import QtCore

from nanoscope.analysis.detect import detect_events
from nanoscope.analysis.events import EventSet
from nanoscope.core.analysis import AnalysisResult
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.panels import StatsPanel

FS = 50_000.0
N = 120_000
I0 = -1000.0


# ================================================================================================
# builders
# ================================================================================================
def _signal(seed=3) -> np.ndarray:
    """Ten events of DELIBERATELY different depths and dwells, so an accepted-only median
    is a different number from an all-rows median."""
    rng = np.random.default_rng(seed)
    x = np.full(N, I0, dtype=np.float64) + rng.normal(0.0, 8.0, N)
    for i in range(10):
        s = 5_000 + i * 10_000
        e = s + 300 + 90 * i
        x[s:e] -= 150.0 + 35.0 * i
    return x.astype(np.float32)


def _doc(tmp_path, signal, units="pA", name="apo.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units=units, kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(signal.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(signal,))


@pytest.fixture
def win(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _open_and_detect(win, tmp_path, units="pA", name="apo.dat"):
    signal = _signal()
    doc_id = win._session.add_document(_doc(tmp_path, signal, units=units, name=name))
    win._create_default_layers(win._session.document(doc_id))
    win._on_selection(doc_id, 0)
    df, prov = detect_events(signal, FS, baseline=I0, min_dwell_s=1e-4,
                             recording=name, pore_id="pore-1")
    win._session.add_result(AnalysisResult(
        module_id="detect.schmitt", module_version="1.0", doc_id=doc_id, channel=0,
        params={}, calibration_fingerprint=win._session.calibration(doc_id, 0).fingerprint(),
        payload=EventSet(df, prov)))
    return doc_id


def _rows(win) -> dict[str, str]:
    """What the user reads: the widget's own vertical header text -> its cell text."""
    table = win._analysis_panel._stats.summary_table
    out = {}
    for i in range(table.rowCount()):
        header = table.verticalHeaderItem(i)
        item = table.item(i, 0)
        out[header.text() if header else ""] = item.text() if item else ""
    return out


def _label(win, key: str) -> str:
    return win._analysis_panel._stats.summary_label(key)


def _value(win, key: str) -> str:
    return win._analysis_panel._stats.summary_text(key)


def _reject_deepest(win, k: int) -> list[int]:
    """Curate through the model call a checkbox click makes."""
    table = win._analysis_panel._table
    df = win._analysis_panel.current_events_df()
    ids = df.nlargest(k, "depth_pA")["event_id"].tolist()
    for eid in ids:
        table.set_accepted_by_id(int(eid), False)
    return [int(i) for i in ids]


# ================================================================================================
# FE-02 — the medians count the rows the label names
# ================================================================================================
def test_the_medians_move_when_the_curator_rejects_events(win, tmp_path):
    """The defect exactly: reject the deepest rows and the depth median must follow."""
    _open_and_detect(win, tmp_path)
    before = {k: _value(win, k) for k in ("median_dwell_us", "median_depth_pA",
                                          "median_frac_depth", "median_baseline_pA")}
    n_before = _value(win, "n_accepted")
    _reject_deepest(win, 4)
    after = {k: _value(win, k) for k in before}

    assert _value(win, "n_accepted") != n_before, "the control row did not move"
    assert after["median_depth_pA"] != before["median_depth_pA"], (
        f"median depth unchanged after rejecting the deepest events: {before} -> {after}")

    df = win._analysis_panel.current_events_df()
    adf = df[df["accepted"].to_numpy(dtype=bool)]
    assert float(after["median_depth_pA"]) == pytest.approx(
        float(np.nanmedian(adf["depth_pA"].to_numpy())), rel=1e-3)
    assert float(after["median_dwell_us"]) == pytest.approx(
        float(np.median(adf["dwell_s"].to_numpy())) * 1e6, rel=1e-3)
    assert float(after["median_baseline_pA"]) == pytest.approx(
        float(np.nanmedian(adf["baseline_pA"].to_numpy())), rel=1e-3)
    assert float(after["median_frac_depth"].split()[0]) == pytest.approx(
        float(np.nanmedian(adf["frac_depth"].to_numpy())), rel=1e-3)


def test_no_median_is_ever_taken_over_a_rejected_row(win, tmp_path):
    """Stronger than "it moved": the value must equal the accepted-only median and must
    NOT equal the all-rows one."""
    _open_and_detect(win, tmp_path)
    _reject_deepest(win, 5)
    df = win._analysis_panel.current_events_df()
    adf = df[df["accepted"].to_numpy(dtype=bool)]
    shown = float(_value(win, "median_depth_pA"))
    assert shown == pytest.approx(float(np.nanmedian(adf["depth_pA"].to_numpy())), rel=1e-3)
    assert shown != pytest.approx(float(np.nanmedian(df["depth_pA"].to_numpy())), rel=1e-6)


def test_every_median_row_names_the_rows_it_counted(win, tmp_path):
    """A label the reader can act on, beside the number — the rule the rate row already
    followed and the four medians beneath it did not."""
    _open_and_detect(win, tmp_path)
    for key in ("median_dwell_us", "median_depth_pA", "median_frac_depth",
                "median_baseline_pA"):
        assert "accepted" in _label(win, key), (key, _label(win, key))
    # ...and the row that is deliberately over ALL rows still says nothing about
    # acceptance, because it counts every detected row.
    assert "accepted" not in _label(win, "n_events")


def test_the_summary_and_the_plots_window_agree_about_the_same_column(win, tmp_path):
    """The cross-surface half of the finding: Plots defaults to "Use accepted events
    only", so its dwell/depth distributions and the Summary's medians of the same
    columns must be over the same rows."""
    _open_and_detect(win, tmp_path)
    _reject_deepest(win, 3)
    win._open_plots()
    plots = win._plots_window
    assert plots._accepted_only(), "Plots still defaults to accepted-only"
    df = win._analysis_panel.current_events_df()
    adf = df[df["accepted"].to_numpy(dtype=bool)]
    drawn = plots._current_frame()
    assert drawn is not None and len(drawn) == len(adf), (
        "the Plots window draws the accepted rows; the Summary must median the same ones")
    assert float(np.nanmedian(drawn["depth_pA"].to_numpy())) == pytest.approx(
        float(_value(win, "median_depth_pA")), rel=1e-3), (
        "one detection, one median — the two surfaces must not answer differently")
    assert float(_value(win, "median_depth_pA")) == pytest.approx(
        float(np.nanmedian(adf["depth_pA"].to_numpy())), rel=1e-3)


def test_rejecting_every_row_leaves_no_median_rather_than_a_median_of_nothing(win, tmp_path):
    _open_and_detect(win, tmp_path)
    df = win._analysis_panel.current_events_df()
    _reject_deepest(win, len(df))
    assert _value(win, "n_accepted") == "0"
    for key in ("median_dwell_us", "median_depth_pA", "median_frac_depth",
                "median_baseline_pA"):
        assert _value(win, key) == "—", (key, _value(win, key))


def test_the_medians_are_still_mutually_consistent(win, tmp_path):
    """depth / |baseline| must still reproduce the dI/I0 beside it — the three are over
    the same rows, before and after curation."""
    _open_and_detect(win, tmp_path)
    _reject_deepest(win, 4)
    depth = float(_value(win, "median_depth_pA"))
    base = float(_value(win, "median_baseline_pA"))
    frac = float(_value(win, "median_frac_depth").split()[0])
    assert depth / abs(base) == pytest.approx(frac, rel=0.05)


# ================================================================================================
# FE-04 — the caption is the channel's unit, never an invented one
# ================================================================================================
def test_a_pA_channel_is_captioned_pA(win, tmp_path):
    _open_and_detect(win, tmp_path, units="pA")
    assert _label(win, "median_depth_pA") == "median depth (pA, accepted)"
    assert _label(win, "median_baseline_pA") == "open pore (median stamped, pA, accepted)"


def test_an_undeclared_channel_is_not_captioned_pA(win, tmp_path):
    """The state every fresh HDF5/CSV channel is in. The PSD window says "units unknown"
    for it; so does this panel now."""
    _open_and_detect(win, tmp_path, units="")
    assert "units unknown" in _label(win, "median_depth_pA")
    assert "units unknown" in _label(win, "median_baseline_pA")
    assert "(pA" not in _label(win, "median_depth_pA")
    assert "(pA" not in _label(win, "median_baseline_pA")
    # The rows are on screen as the widget's own header text, not merely in a template.
    labels = list(_rows(win))
    assert any("units unknown" in lbl for lbl in labels), labels


def test_declaring_the_units_in_calibration_re_captions_the_summary(win, tmp_path):
    """A units-only edit does not touch the array ("target_units is a label"), so the
    numbers on screen ARE nA the moment the user says so — and the caption follows at
    once, instead of asserting pA until the next detection."""
    doc_id = _open_and_detect(win, tmp_path, units="")
    before = _value(win, "median_depth_pA")
    win._session.set_calibration(doc_id, 0,
                                 ChannelCalibration(gain=1.0, offset=0.0, target_units="nA"),
                                 write_sidecar=False)
    assert _label(win, "median_depth_pA") == "median depth (nA, accepted)"
    assert _label(win, "median_baseline_pA") == "open pore (median stamped, nA, accepted)"
    assert _value(win, "median_depth_pA") == before, (
        "a label edit must not change the numbers")


def test_a_gain_edit_does_not_re_caption_numbers_measured_under_the_old_gain(win, tmp_path):
    """The measurement table's rule, applied here: a stamp saying what the numbers were
    measured under must not be rewritten by a later edit. The gain moved, so the depths
    above are the OLD gain's — the staleness banner says so and the caption stands
    until a re-run re-measures them."""
    doc_id = _open_and_detect(win, tmp_path, units="pA")
    shown = _value(win, "median_depth_pA")
    win._session.set_calibration(doc_id, 0,
                                 ChannelCalibration(gain=2.0, offset=0.0, target_units="nA"),
                                 write_sidecar=False)
    assert _label(win, "median_depth_pA") == "median depth (pA, accepted)", (
        "the caption was rewritten under a gain the numbers were never measured under")
    assert _value(win, "median_depth_pA") == shown
    banner = win._analysis_panel._stale
    # isHidden(), not isVisibleTo(panel): the stale line lives in the Results dock now, so the
    # panel is not its ancestor and isVisibleTo would depend on the window being shown.
    assert not banner.isHidden()
    assert "Calibration changed" in banner.message()


def test_repointing_to_another_recording_does_not_carry_the_units_across(win, tmp_path):
    """A label left standing from the previous recording is a wrong label on the next
    run's numbers — the same failure the table's clock offset is cleared for."""
    _open_and_detect(win, tmp_path, units="pA", name="first.dat")
    assert "(pA" in _label(win, "median_depth_pA")
    signal = _signal(seed=9)
    other = win._session.add_document(_doc(tmp_path, signal, units="", name="second.dat"))
    win._create_default_layers(win._session.document(other))
    win._on_selection(other, 0)
    assert "units unknown" in _label(win, "median_depth_pA"), _label(win, "median_depth_pA")
    assert _value(win, "median_depth_pA") == "—"


def test_the_row_tooltip_explains_both_the_rows_and_the_unit(win, tmp_path):
    _open_and_detect(win, tmp_path, units="")
    tip = win._analysis_panel._stats.summary_tooltip("median_depth_pA")
    assert "ACCEPTED" in tip
    assert "units unknown" in tip


def test_a_bare_stats_panel_declares_no_units_it_has_not_been_given(qapp, qtbot):
    """Constructed with no channel behind it, the panel must not assert pA."""
    panel = StatsPanel()
    qtbot.addWidget(panel)
    assert "units unknown" in panel.summary_label("median_depth_pA")
    panel.set_units("nS")
    assert panel.summary_label("median_depth_pA") == "median depth (nS, accepted)"
    panel.set_units("")
    assert "units unknown" in panel.summary_label("median_depth_pA")


def test_set_units_keeps_the_row_header_tooltips(qapp, qtbot):
    """setVerticalHeaderLabels replaces the header items; the explanations must survive,
    or hovering a row name stops explaining it."""
    panel = StatsPanel()
    qtbot.addWidget(panel)
    panel.set_units("nA")
    table = panel.summary_table
    keys = [k for _lbl, k in StatsPanel._SUMMARY_ROWS]
    i = keys.index("median_depth_pA")
    assert "ACCEPTED" in table.verticalHeaderItem(i).toolTip()
