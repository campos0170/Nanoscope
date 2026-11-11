"""The Noise page's Signal choice: the raw calibrated trace, or the processed one.

*Processed* is ``conditioned_trace`` of the calibrated channel under the Detection panel's
Preprocess form (the array Measure's "Processed signal" and Export processed trace build,
with Normalize left out), run over the whole recording and sliced to the analysed window.
It is a compute setting: pending until Update plot, saved with the figure, and the result
says which signal it is. Raw is the default and is the page exactly as it was. Synthetic
data only; offscreen.
"""
from __future__ import annotations

import json
import os
import warnings
import zipfile

import nanoscope  # noqa: F401
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from copy import deepcopy
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.analysis import allpoints as ap_mod
from nanoscope.analysis import psd as psd_mod
from nanoscope.analysis.baseline import collect_baseline_warnings
from nanoscope.analysis.conditioning import conditioned_trace
from nanoscope.analysis.conditioning_recipes import SCOPE_SENTENCE
from nanoscope.core.analysis_workspace import capture_psd, restore_psd
from nanoscope.core.figure_codecs import decode_legacy, encode_legacy, merge_native_capture
from nanoscope.core.project import (AnalyticalWorkspace, DocumentRef, ProjectState,
                                    load_project, save_project)
from nanoscope.core.sample_rate import SampleRateDeclaration
from nanoscope.core.session import Session
from nanoscope.gui import psd_panel
from nanoscope.gui.figure_adapter_pages import capture_applied_record
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.psd_panel import (VIEW_ALLPOINTS, VIEW_FFT, VIEW_PSD, VIEW_SPECTROGRAM,
                                     PSDWindow)

from . import p1_hosting as H
from .test_p0_reset_on_quantity_change import _pick
from .test_plot_window_routes import _edit_double_click, _main_window, _noisy_csv
from .test_s3_units_psd_window import _doc

LOWPASS = {"filter_cutoff_hz": 2000.0, "filter_order": 8, "filter_kind": "bessel"}
LOWPASS_TEXT = "Low-pass Bessel 8 · 2 kHz"
NORMALIZE = {"normalize_enabled": True, "normalize_mode": "zscore"}
DRIFT = {"detrend_enabled": True, "detrend_method": "block_median", "detrend_window_s": 0.5}
PROCESSED = "processed"
RAW = "raw"
SIDE_VIEWS = ((VIEW_FFT, "FFT"), (VIEW_SPECTROGRAM, "Spectrogram"),
              (VIEW_ALLPOINTS, "All-points histogram"))
BASIS_COLUMNS = {"signal_basis", "signal_chain", "signal_chain_params"}


def _page(qtbot, session, doc_id, chain=None, *, managed=True):
    """A Noise page pointed at ``doc_id`` whose Preprocess form is the returned dict."""
    page = PSDWindow(session, defer_initial_refresh=True)
    qtbot.addWidget(page)
    page._nanoscope_figure_managed = managed
    form = dict(chain or {})
    page.set_processing_provider(lambda: form)
    if doc_id is not None:
        page.set_target(doc_id, 0)
    return page, form


def _update(page):
    assert page.inspector_adapter().update() is not False


def _samples(session, doc_id):
    chan = session.calibrated(doc_id, 0)
    return np.asarray(chan.data), float(chan.sample_rate_hz)


def _welch(page, seg, fs):
    nps, nov = page._resolve_welch_params(seg.size)
    return psd_mod.welch_psd(seg, fs, nperseg=nps, noverlap=nov,
                             window=page._window_combo.currentText())


def _count(monkeypatch, owner, name, calls):
    original = getattr(owner, name)

    def counted(*args, **kwargs):
        calls.append(name)
        return original(*args, **kwargs)
    monkeypatch.setattr(owner, name, counted)


def _band(freqs, psd, lo, hi):
    return float(np.mean(psd[(freqs >= lo) & (freqs <= hi)]))


def _kept(page):
    """The recordings whose whole-recording processed trace the page holds."""
    return sorted(key[0] for key in page._processed_traces)


def _side_view_result(page, view):
    return {VIEW_FFT: lambda: page.fft_spectrum().mag,
            VIEW_SPECTROGRAM: lambda: page.spectrogram().psd,
            VIEW_ALLPOINTS: lambda: page.all_points().counts}[view]()


def _side_view_kernels(page, samples, fs):
    """What each side view computes from ``samples`` under the page's default options."""
    return {
        VIEW_FFT: psd_mod.fft_magnitude(samples, fs, window="hann").mag,
        VIEW_SPECTROGRAM: psd_mod.spectrogram_psd(
            samples, fs, nperseg=None, window="hann",
            max_columns=psd_panel._SPEC_MAX_COLUMNS).psd,
        VIEW_ALLPOINTS: ap_mod.all_points_histogram(
            samples, bins=int(page._hist_bins_combo.currentText()), clip_quantile=0.0).counts,
    }


@pytest.fixture
def one(qapp, qtbot, tmp_path):
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, "A.dat", units="pA"))
    return session, doc_id


# -- Raw: the default, and the page as it was -----------------------------------------------------

def test_raw_is_the_default_and_is_the_calibrated_traces_own_welch_estimate(
        one, qtbot, monkeypatch):
    session, doc_id = one
    calls = []
    _count(monkeypatch, psd_panel, "conditioned_trace", calls)
    page, _form = _page(qtbot, session, doc_id, LOWPASS)
    assert page._signal_combo.currentData() == RAW
    assert [page._signal_combo.itemText(i) for i in range(page._signal_combo.count())] == [
        "Raw (calibrated)", "Processed (Preprocess chain)"]
    assert page._signal_combo.toolTip() == (
        "Raw: the calibrated trace.\n"
        "Processed: the trace after the Detection panel's Preprocess chain as it is set now, "
        "applied to every compared recording. Normalize is left out.")
    assert page._recording_group.isAncestorOf(page._signal_combo)
    data, fs = _samples(session, doc_id)
    freqs, psd = _welch(page, data, fs)
    np.testing.assert_array_equal(page.psd_result()[0], freqs)
    np.testing.assert_array_equal(page.psd_result()[1], psd)
    assert calls == [] and page.signal_basis() == RAW and page.signal_chain() == ""
    # Nothing a raw page shows or saves mentions the choice.
    assert page._plot.titleLabel.text == "PSD — A.dat"
    assert "Signal" not in page.details_text() and "processed" not in page.details_text()
    assert not BASIS_COLUMNS & set(page._plot_data().columns)
    payload = capture_psd(page).payload
    assert "signal" not in payload and "signal_basis" not in payload["config"]
    assert payload["pending_analysis"] is None
    assert "signal_basis" not in page.template_settings()[0]
    assert page._processed_traces == {}


# -- Processed: the shared trace, over the whole recording, then the window ------------------------

def test_processed_waits_for_update_then_is_the_welch_estimate_of_the_shared_processed_trace(
        one, qtbot, monkeypatch):
    session, doc_id = one
    page, _form = _page(qtbot, session, doc_id, LOWPASS)
    page._start_spin.setValue(0.5)
    page._end_spin.setValue(1.5)
    _update(page)
    raw_psd = page.psd_result()[1]
    before = page._plot_data()
    calls = []
    _count(monkeypatch, psd_panel.psd_mod, "welch_psd", calls)
    _count(monkeypatch, psd_panel, "conditioned_trace", calls)
    heard = []
    page.pendingChanged.connect(lambda: heard.append(1))

    page.set_signal_basis(PROCESSED)
    state = page.inspector_adapter().pending()
    assert heard and state.count == 1 and state.reasons == ("signal basis",)
    assert calls == [], "the choice is a draft until Update plot"
    np.testing.assert_array_equal(page.psd_result()[1], raw_psd)
    pd.testing.assert_frame_equal(page._plot_data(), before)
    assert page.signal_basis() == RAW

    _update(page)
    assert calls == ["conditioned_trace", "welch_psd"]
    assert page.inspector_adapter().pending().count == 0
    assert page.signal_basis() == PROCESSED and page.signal_chain() == LOWPASS_TEXT
    data, fs = _samples(session, doc_id)
    i0, i1 = int(0.5 * fs), int(1.5 * fs)
    whole = conditioned_trace(data, fs, LOWPASS)
    freqs, psd = _welch(page, whole[i0:i1], fs)
    np.testing.assert_array_equal(page.psd_result()[0], freqs)
    np.testing.assert_array_equal(page.psd_result()[1], psd)
    # The chain ran on the recording, not on the cut: a cut filtered alone starts cold.
    alone = _welch(page, conditioned_trace(data[i0:i1], fs, LOWPASS), fs)[1]
    assert not np.array_equal(page.psd_result()[1], alone)
    # Above the 2 kHz low-pass the noise is gone; below it, it is what it was.
    raw_freqs, raw = _welch(page, data[i0:i1], fs)
    assert _band(freqs, psd, 10_000, 20_000) < 1e-6 * _band(raw_freqs, raw, 10_000, 20_000)
    assert _band(freqs, psd, 50, 500) == pytest.approx(_band(raw_freqs, raw, 50, 500), rel=0.05)
    assert page.curves()[0].units == "pA" and page._plot.getAxis("left").labelText == "PSD (pA²/Hz)"
    assert _kept(page) == [doc_id]

    # Back to Raw is a draft too; Update returns the raw curve and lets the trace go.
    page.set_signal_basis(RAW)
    assert page.inspector_adapter().pending().reasons == ("signal basis",)
    np.testing.assert_array_equal(page.psd_result()[1], psd)
    _update(page)
    np.testing.assert_array_equal(page.psd_result()[1], raw_psd)
    assert page.signal_basis() == RAW and page._processed_traces == {}


def test_all_four_views_are_views_of_the_chosen_signal(one, qtbot, monkeypatch):
    session, doc_id = one
    page, _form = _page(qtbot, session, doc_id, LOWPASS)
    data, fs = _samples(session, doc_id)
    whole = np.asarray(conditioned_trace(data, fs, LOWPASS))
    calls = []
    _count(monkeypatch, psd_panel, "conditioned_trace", calls)
    page.set_signal_basis(PROCESSED)
    assert not page._hist_clip_check.isChecked()
    for view, _head in SIDE_VIEWS:
        page.set_view(view)
        _update(page)
        assert page.current_view() == view
    # One pass over the recording served the spectrum and all three views.
    assert calls == ["conditioned_trace"]
    np.testing.assert_array_equal(page.psd_result()[1], _welch(page, whole, fs)[1])
    expected = _side_view_kernels(page, whole, fs)
    for view, head in SIDE_VIEWS:
        np.testing.assert_array_equal(_side_view_result(page, view), expected[view])
        assert page._view_titles[view] == f"{head} — A.dat (processed: {LOWPASS_TEXT})"
        assert f"Signal: processed — {LOWPASS_TEXT}." in page.caption_text(view)
    assert not np.array_equal(page.fft_spectrum().mag,
                              psd_mod.fft_magnitude(data, fs, window="hann").mag)
    # Raw again: the cached processed views are not served under a raw name.
    page.set_signal_basis(RAW)
    _update(page)
    np.testing.assert_array_equal(page.all_points().counts,
                                  _side_view_kernels(page, data, fs)[VIEW_ALLPOINTS])
    assert page._view_titles[VIEW_ALLPOINTS] == "All-points histogram — A.dat"
    assert "Signal" not in page.caption_text(VIEW_ALLPOINTS)


# -- the result says what it is -----------------------------------------------------------------

def test_the_title_the_details_and_the_csv_name_the_basis(one, qtbot, tmp_path):
    session, doc_id = one
    page, _form = _page(qtbot, session, doc_id, LOWPASS)
    page.set_signal_basis(PROCESSED)
    _update(page)
    assert page._plot.titleLabel.text == f"PSD — A.dat (processed: {LOWPASS_TEXT})"
    details = page.details_text().splitlines()
    assert details[0] == f"Signal: processed — {LOWPASS_TEXT}"
    assert details[1] == "The Detection panel's Preprocess chain, applied to every curve."
    out = page.export_data(str(tmp_path / "noise.csv"))
    table = pd.read_csv(out)
    assert set(table["signal_basis"]) == {PROCESSED}
    assert set(table["signal_chain"]) == {LOWPASS_TEXT}
    assert [json.loads(text) for text in set(table["signal_chain_params"])] == [LOWPASS]
    assert table["details"].iloc[0].startswith(f"Signal: processed — {LOWPASS_TEXT}")
    payload = capture_psd(page).payload
    assert payload["config"]["signal_basis"] == PROCESSED
    assert payload["signal"]["basis"] == PROCESSED and payload["signal"]["chain"] == LOWPASS_TEXT
    assert json.loads(payload["signal"]["chain_params"]) == LOWPASS
    # A chain too long for a title is counted there and spelled out in the details.
    long_chain = {**LOWPASS, "notch_enabled": True, "notch_freq_hz": 60.0, "notch_harmonics": 3,
                  "notch_q": 30.0, "rmean_enabled": True, "rmean_window_s": 1e-4}
    other, _ = _page(qtbot, session, doc_id, long_chain)
    other.set_signal_basis(PROCESSED)
    _update(other)
    assert other._plot.titleLabel.text == "PSD — A.dat (processed: 3 stages)"
    assert other.details_text().splitlines()[0] == (
        "Signal: processed — Notch 60/120/180 Hz · Q 30 → Running mean 0.1 ms → "
        + LOWPASS_TEXT)


def test_every_views_csv_names_the_basis_the_chain_and_its_parameters(one, qtbot):
    """Finding 11: the chain in words and exactly, on the export of each of the four views;
    Normalize is no part of the parameters (it was left out of the signal)."""
    session, doc_id = one
    form = {**LOWPASS, **NORMALIZE, "notch_enabled": False, "notch_freq_hz": 60.0}
    page, _form = _page(qtbot, session, doc_id, form)
    page.set_signal_basis(PROCESSED)
    chain_text = LOWPASS_TEXT + " · Normalize left out"
    used = {**LOWPASS, "notch_enabled": False, "notch_freq_hz": 60.0}
    for view in (VIEW_PSD, VIEW_FFT, VIEW_SPECTROGRAM, VIEW_ALLPOINTS):
        page.set_view(view)
        _update(page)
        frame = page._plot_data()
        assert BASIS_COLUMNS <= set(frame.columns), view
        named = frame[frame["signal_basis"].notna()]
        # The spectrogram's CSV carries its metadata on one row, by that file's design.
        assert len(named) == (1 if view == VIEW_SPECTROGRAM else len(frame)), view
        assert set(named["signal_basis"]) == {PROCESSED}, view
        assert set(named["signal_chain"]) == {chain_text}, view
        params = set(named["signal_chain_params"])
        assert len(params) == 1 and json.loads(params.pop()) == used, view
        columns = list(frame.columns)
        assert columns.index("signal_chain_params") == columns.index("signal_chain") + 1
    # The same four exports of a raw result carry none of the three columns.
    page.set_signal_basis(RAW)
    for view in (VIEW_PSD, VIEW_FFT, VIEW_SPECTROGRAM, VIEW_ALLPOINTS):
        page.set_view(view)
        _update(page)
        assert not BASIS_COLUMNS & set(page._plot_data().columns), view


def test_a_typed_title_keeps_the_basis_marker_on_every_view_and_in_the_exported_figure(
        one, qtbot, tmp_path):
    """Finding 4: the marker is not the title's author's to remove."""
    session, doc_id = one
    page, _form = _page(qtbot, session, doc_id, LOWPASS)
    page.set_title("Pore 7")
    assert page._plot.titleLabel.text == "Pore 7", "a raw result's typed title is untouched"
    page.set_signal_basis(PROCESSED)
    _update(page)
    marked = f"Pore 7 (processed: {LOWPASS_TEXT})"
    assert page._plot.titleLabel.text == marked
    out = page.save_image(str(tmp_path / "typed.svg"))
    assert out is not None and "Pore 7 (processed: Low-pass Bessel 8" in out.read_text(
        encoding="utf-8")
    plots = {VIEW_FFT: page._fft_plot, VIEW_SPECTROGRAM: page._spec_plot,
             VIEW_ALLPOINTS: page._hist_plot}
    for view, _head in SIDE_VIEWS:
        page.set_view(view)
        _update(page)
        assert plots[view].titleLabel.text == marked, view
    # Typed again, the title is still followed by the marker; cleared, it is automatic.
    page.set_view(VIEW_PSD)
    _update(page)
    page.set_title("Run 3")
    assert page._plot.titleLabel.text == f"Run 3 (processed: {LOWPASS_TEXT})"
    # An automatic title edited in place already ends with the marker: it is not doubled.
    page.set_title(f"Quiet stretch (processed: {LOWPASS_TEXT})")
    assert page._plot.titleLabel.text == f"Quiet stretch (processed: {LOWPASS_TEXT})"
    page.set_title("")
    assert page._plot.titleLabel.text == f"PSD — A.dat (processed: {LOWPASS_TEXT})"
    # Raw again: the typed title is exactly what was typed.
    page.set_title("Pore 7")
    page.set_signal_basis(RAW)
    _update(page)
    assert page._plot.titleLabel.text == "Pore 7"
    raw_svg = page.save_image(str(tmp_path / "raw.svg")).read_text(encoding="utf-8")
    assert "Pore 7" in raw_svg and "processed" not in raw_svg


# -- freshness: a chain edited under a processed result --------------------------------------------

def test_a_chain_edit_makes_a_processed_result_pending_and_leaves_a_raw_one_alone(
        one, qtbot, monkeypatch):
    session, doc_id = one
    page, form = _page(qtbot, session, doc_id, LOWPASS)
    judged, edits, shown = [], [], []
    page.chainJudgementChanged.connect(lambda: judged.append(1))
    page.pendingChanged.connect(lambda: edits.append(1))
    page.inspector_adapter().pending_changed.connect(lambda: shown.append(1))
    calls = []
    _count(monkeypatch, psd_panel.psd_mod, "welch_psd", calls)
    _count(monkeypatch, psd_panel, "conditioned_trace", calls)
    # Raw: the form may move as it likes.
    form["filter_cutoff_hz"] = 5000.0
    assert page.processing_chain_changed() is False
    assert judged == [] and page.inspector_adapter().pending().count == 0

    page.set_signal_basis(PROCESSED)
    _update(page)
    drawn = page.psd_result()[1]
    assert page.signal_chain() == "Low-pass Bessel 8 · 5 kHz"
    judged.clear(), edits.clear(), shown.clear(), calls.clear()
    form["filter_cutoff_hz"] = 1000.0
    assert page.processing_chain_changed() is True
    state = page.inspector_adapter().pending()
    assert judged == [1] and state.count == 1 and state.reasons == ("Preprocess chain changed",)
    # A judgement is not an edit (finding 5): the Update button hears it, the signal that
    # marks a figure as changed is not sent.
    assert shown == [1] and edits == []
    assert page.processing_chain_changed() is False
    assert judged == [1], "said once, not on every form signal"
    assert calls == [], "nothing is recomputed until Update plot"
    np.testing.assert_array_equal(page.psd_result()[1], drawn)
    assert page.signal_chain() == "Low-pass Bessel 8 · 5 kHz"       # what is drawn, still
    # A knob of a stage that is off changes no sample, so it is no change.
    form["filter_cutoff_hz"] = 5000.0
    form["notch_freq_hz"] = 50.0
    assert page.processing_chain_changed() is True
    assert judged == [1, 1] and page.inspector_adapter().pending().count == 0
    form["filter_cutoff_hz"] = 1000.0
    page.processing_chain_changed()
    assert edits == []
    _update(page)
    assert calls == ["conditioned_trace", "welch_psd"]
    assert page.inspector_adapter().pending().count == 0
    assert page.signal_chain() == "Low-pass Bessel 8 · 1 kHz"
    assert not np.array_equal(page.psd_result()[1], drawn)


# -- Normalize is left out: the spectrum stays in the channel's units --------------------------------

def test_normalize_is_left_out_of_the_processed_signal_and_of_its_freshness(
        one, qtbot, monkeypatch):
    session, doc_id = one
    seen = []
    real = psd_panel.conditioned_trace
    monkeypatch.setattr(psd_panel, "conditioned_trace",
                        lambda raw, fs, params: (seen.append(dict(params)), real(raw, fs, params))[1])
    page, form = _page(qtbot, session, doc_id, {**LOWPASS, **NORMALIZE})
    page.set_signal_basis(PROCESSED)
    _update(page)
    # The chain ran without the stage, so the numbers are the low-pass's alone, in pA.
    assert seen == [LOWPASS]
    data, fs = _samples(session, doc_id)
    np.testing.assert_array_equal(
        page.psd_result()[1], _welch(page, np.asarray(conditioned_trace(data, fs, LOWPASS)), fs)[1])
    assert not np.array_equal(
        page.psd_result()[1],
        _welch(page, np.asarray(conditioned_trace(data, fs, {**LOWPASS, **NORMALIZE})), fs)[1])
    assert page.curves()[0].units == "pA" and page._plot.getAxis("left").labelText == "PSD (pA²/Hz)"
    chain_text = f"{LOWPASS_TEXT} · Normalize left out"
    assert page.signal_basis() == PROCESSED and page.signal_chain() == chain_text
    assert page._plot.titleLabel.text == f"PSD — A.dat (processed: {chain_text})"
    assert page.details_text().splitlines()[0] == f"Signal: processed — {chain_text}"
    assert ("NOTE  Normalize is left out here, so the spectrum stays in pA."
            in page.details_text().splitlines())
    saved = capture_psd(page).payload
    assert saved["signal"]["chain"] == chain_text
    assert json.loads(saved["signal"]["chain_params"]) == LOWPASS
    assert "Normalize is left out here, so the spectrum stays in pA." in saved["compute_notes"]

    # The Normalize settings are no part of the chain: editing them marks nothing stale,
    # and an Update afterwards reads the kept trace instead of running the chain again.
    seen.clear()
    for edit in ({"normalize_mode": "robust"}, {"normalize_clip_sigma": 3.0},
                 {"normalize_enabled": False}):
        form.update(edit)
        assert page.processing_chain_changed() is False
        assert page.inspector_adapter().pending().count == 0
    _update(page)
    assert seen == [] and page.signal_chain() == LOWPASS_TEXT
    assert "Normalize" not in page.details_text()
    # A chain counted in the title keeps the words.
    long_chain = {**LOWPASS, **NORMALIZE, "notch_enabled": True, "notch_freq_hz": 60.0,
                  "notch_harmonics": 3, "notch_q": 30.0, "rmean_enabled": True,
                  "rmean_window_s": 1e-4}
    other, _ = _page(qtbot, session, doc_id, long_chain)
    other.set_signal_basis(PROCESSED)
    _update(other)
    assert other._plot.titleLabel.text == (
        "PSD — A.dat (processed: 3 stages · Normalize left out)")
    # Normalize alone: nothing is left to run, and both notes say what the result is.
    alone, _ = _page(qtbot, session, doc_id, dict(NORMALIZE))
    raw_psd = alone.psd_result()[1]
    alone.set_signal_basis(PROCESSED)
    _update(alone)
    np.testing.assert_array_equal(alone.psd_result()[1], raw_psd)
    assert alone.signal_chain() == "no stage enabled · Normalize left out"
    notes = [line for line in alone.details_text().splitlines() if line.startswith("NOTE  ")]
    assert notes == [
        "NOTE  No Preprocess stage is enabled, so Processed is the same as Raw. Measure and "
        "Export may still use the stored detection's chain.",
        "NOTE  Normalize is left out here, so the spectrum stays in pA."]


def test_normalize_left_out_names_no_unit_the_channel_does_not_declare(qapp, qtbot, tmp_path):
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, "pore.h5"))            # units undeclared
    session.add_document(_doc(tmp_path, "A.dat", units="pA", seed=2))
    page, _form = _page(qtbot, session, doc_id, {**LOWPASS, **NORMALIZE})
    page.set_signal_basis(PROCESSED)
    _update(page)
    assert ("NOTE  Normalize is left out here, so the spectrum stays in the channel's units."
            in page.details_text().splitlines())
    # Two curves that do not share one declared unit: each keeps its own.
    page._compare_list.item(0).setCheckState(QtCore.Qt.Checked)
    _update(page)
    assert ("NOTE  Normalize is left out here, so the spectrum stays in each channel's own "
            "units." in page.details_text().splitlines())


# -- units and failures: never the raw trace under a processed name --------------------------------

def test_no_enabled_stage_is_the_raw_trace_and_says_so(one, qtbot):
    session, doc_id = one
    chain = {"filter_cutoff_hz": None, "notch_enabled": False, "notch_freq_hz": 60.0}
    page, _form = _page(qtbot, session, doc_id, chain)
    raw_psd = page.psd_result()[1]
    page.set_signal_basis(PROCESSED)
    _update(page)
    data, fs = _samples(session, doc_id)
    np.testing.assert_array_equal(np.asarray(conditioned_trace(data, fs, chain)), data)
    np.testing.assert_array_equal(page.psd_result()[1], raw_psd)
    assert page.signal_basis() == PROCESSED and page.signal_chain() == "no stage enabled"
    assert page._plot.titleLabel.text == "PSD — A.dat (processed: no stage enabled)"
    assert ("NOTE  No Preprocess stage is enabled, so Processed is the same as Raw. Measure "
            "and Export may still use the stored detection's chain." in page.details_text())
    assert page._processed_traces == {}, "the raw samples are not copied to be kept"


def test_a_refused_update_leaves_the_picture_in_place_and_says_why(qapp, qtbot, tmp_path,
                                                                   monkeypatch):
    """Finding 7: every Processed refusal is settled before anything is cleared."""
    session = Session()
    fast = session.add_document(_doc(tmp_path, "A.dat", units="pA"))
    slow_doc = _doc(tmp_path, "B.dat", units="pA", seed=1)
    session.add_document(replace(slow_doc, meta=replace(slow_doc.meta, sample_rate_hz=10_000.0)))
    # 8 kHz is below A's Nyquist (25 kHz) and above B's (5 kHz).
    chain = {**LOWPASS, "filter_cutoff_hz": 8000.0}
    page, form = _page(qtbot, session, fast, chain)
    page._compare_list.item(0).setCheckState(QtCore.Qt.Checked)
    page.set_signal_basis(PROCESSED)
    _update(page)
    # A compared recording that cannot be built is left out, with its reason.
    assert page.curve_labels() == ["A.dat"]
    note = next(line for line in page.details_text().splitlines() if line.startswith("NOTE  B.dat"))
    assert note.startswith("NOTE  B.dat: the Preprocess chain could not be applied: ")
    assert page.signal_basis() == PROCESSED and _kept(page) == [fast]

    # The primary itself, on a page that shows a result: the Update is refused and the
    # picture, its numbers, its kept trace and the pending state stay as they were.
    drawn, details, title = page.psd_result()[1], page.details_text(), page._plot.titleLabel.text
    kept = dict(page._processed_traces)
    form["filter_cutoff_hz"] = 40_000.0
    page.processing_chain_changed()
    calls = []
    _count(monkeypatch, psd_panel.psd_mod, "welch_psd", calls)
    assert page.inspector_adapter().update() is False
    assert calls == [], "nothing was computed"
    assert page.statusBar().currentMessage().startswith(
        "A.dat: the Preprocess chain could not be applied: ")
    assert page.has_data() and page.curve_labels() == ["A.dat"]
    np.testing.assert_array_equal(page.psd_result()[1], drawn)
    assert page.details_text() == details and page._plot.titleLabel.text == title
    assert page.signal_chain() == "Low-pass Bessel 8 · 8 kHz" and page._message_item is None
    assert page._processed_traces.keys() == kept.keys()
    assert page.inspector_adapter().pending().reasons == ("Preprocess chain changed",)
    assert "8 kHz" in capture_psd(page).payload["signal"]["chain"]
    # Put right, the same press computes.
    form["filter_cutoff_hz"] = 4000.0
    _update(page)
    assert page.signal_chain() == "Low-pass Bessel 8 · 4 kHz"
    assert not np.array_equal(page.psd_result()[1], drawn)

    # No form to read at all (a page nobody connected): the same refusal, picture kept.
    page.set_processing_provider(None)
    drawn = page.psd_result()[1]
    page._start_spin.setValue(0.25)
    assert page.inspector_adapter().update() is False
    assert page.statusBar().currentMessage() == (
        "A.dat: the Preprocess chain is not available, so the processed signal cannot be built.")
    np.testing.assert_array_equal(page.psd_result()[1], drawn)
    assert "start s" in page.inspector_adapter().pending().reasons

    # A page with nothing drawn yet has no picture to keep: the reason is its message.
    empty, _ = _page(qtbot, session, None, {**LOWPASS, "filter_cutoff_hz": 40_000.0})
    empty.set_target(fast, 0, recompute=False)
    empty.set_signal_basis(PROCESSED)
    _update(empty)
    assert not empty.has_data() and empty.curves() == []
    assert empty.details_text().startswith("A.dat: the Preprocess chain could not be applied: ")
    assert empty._message_item.toPlainText() == empty.details_text()
    assert "signal" not in capture_psd(empty).payload
    # Standing alone (no Update button, so no draft to keep apart from the picture) the
    # controls are the truth, and the reason replaces the result as it always did.
    bare = PSDWindow(session, defer_initial_refresh=True)
    qtbot.addWidget(bare)
    bare.set_target(fast, 0)
    assert bare.has_data()
    bare.set_signal_basis(PROCESSED)
    assert not bare.has_data()
    assert bare.details_text() == ("A.dat: the Preprocess chain is not available, so the "
                                   "processed signal cannot be built.")


def test_the_common_chain_failures_are_plain_sentences(qapp, qtbot, tmp_path, monkeypatch):
    """Finding 12: too short for the filter, and out of memory."""
    session = Session()
    doc = _doc(tmp_path, "short.dat", units="pA")
    short = replace(doc, channels=(np.array(doc.channels[0][:20], copy=True),),
                    meta=replace(doc.meta, n_samples=20))
    short_id = session.add_document(short)
    page, _form = _page(qtbot, session, short_id, LOWPASS)
    page.set_signal_basis(PROCESSED)
    assert page.inspector_adapter().update() is False
    assert page.statusBar().currentMessage() == (
        "short.dat: this recording is too short for the Preprocess chain's filters.")
    assert page.signal_basis() == RAW and page.has_data()

    whole_id = session.add_document(_doc(tmp_path, "A.dat", units="pA"))
    other, _ = _page(qtbot, session, whole_id, LOWPASS)

    def out_of_memory(raw, fs, params):
        raise MemoryError()
    monkeypatch.setattr(psd_panel, "conditioned_trace", out_of_memory)
    other.set_signal_basis(PROCESSED)
    assert other.inspector_adapter().update() is False
    assert other.statusBar().currentMessage() == (
        "A.dat: not enough memory to run the Preprocess chain over this recording.")
    # Any other refusal keeps the kernel's own words.
    monkeypatch.setattr(psd_panel, "conditioned_trace",
                        lambda raw, fs, params: (_ for _ in ()).throw(ValueError("odd taps")))
    assert other.inspector_adapter().update() is False
    assert other.statusBar().currentMessage() == (
        "A.dat: the Preprocess chain could not be applied: odd taps")


def test_the_preprocess_scope_sentence_names_the_noise_page():
    assert SCOPE_SENTENCE == (
        "Used by Schmitt/CUSUM detection, Measure ‘Processed signal’, Export processed trace "
        "and the Noise page's Processed signal — not by the ML detector (ML tab ▸ Model "
        "preprocessing).")


# -- the cost of the whole-recording chain ---------------------------------------------------------

def test_a_recording_over_the_size_limit_is_refused_before_anything_is_cleared(
        qapp, qtbot, tmp_path, monkeypatch):
    """Finding 3: the guard is per recording, and it comes before the chain and the clear."""
    assert psd_panel._PROCESSED_MAX_SAMPLES == 30_000_000
    assert psd_panel._too_long_reason("A.dat", 45_200_000) == (
        "A.dat: Processed needs the Preprocess chain run over the whole recording, and this "
        "one is too long for the Noise page (45 M samples; the limit is 30 M). Use the "
        "Preprocess workspace's Spectrum preview on a window instead.")
    session = Session()
    long_id = session.add_document(_doc(tmp_path, "A.dat", units="pA"))          # 100 000
    doc = _doc(tmp_path, "B.dat", units="pA", seed=2)
    brief_id = session.add_document(replace(
        doc, channels=(np.array(doc.channels[0][:40_000], copy=True),),
        meta=replace(doc.meta, n_samples=40_000)))
    monkeypatch.setattr(psd_panel, "_PROCESSED_MAX_SAMPLES", 50_000)
    calls = []
    _count(monkeypatch, psd_panel, "conditioned_trace", calls)

    page, _form = _page(qtbot, session, long_id, LOWPASS)
    drawn, details = page.psd_result()[1], page.details_text()
    page.set_signal_basis(PROCESSED)
    assert page.inspector_adapter().update() is False
    assert page.statusBar().currentMessage() == psd_panel._too_long_reason("A.dat", 100_000)
    assert calls == [], "the chain is not tried on a recording over the limit"
    np.testing.assert_array_equal(page.psd_result()[1], drawn)
    assert page.details_text() == details and page.signal_basis() == RAW
    assert page.inspector_adapter().pending().reasons == ("signal basis",)
    # A recording exactly at the limit is not over it.
    monkeypatch.setattr(psd_panel, "_PROCESSED_MAX_SAMPLES", 100_000)
    _update(page)
    assert calls == ["conditioned_trace"] and page.signal_basis() == PROCESSED

    # As a compared recording it is left out with the same sentence; the primary computes.
    monkeypatch.setattr(psd_panel, "_PROCESSED_MAX_SAMPLES", 50_000)
    calls.clear()
    other, _ = _page(qtbot, session, brief_id, LOWPASS)
    other._compare_list.item(0).setCheckState(QtCore.Qt.Checked)
    other.set_signal_basis(PROCESSED)
    _update(other)
    assert calls == ["conditioned_trace"] and other.curve_labels() == ["B.dat"]
    assert ("NOTE  " + psd_panel._too_long_reason("A.dat", 100_000)
            in other.details_text().splitlines())
    # With no stage enabled there is no chain to run, so there is nothing to refuse.
    idle, _ = _page(qtbot, session, long_id, {"filter_cutoff_hz": None})
    idle.set_signal_basis(PROCESSED)
    _update(idle)
    assert idle.signal_basis() == PROCESSED and idle.signal_chain() == "no stage enabled"


def test_the_chain_runs_once_per_recording_and_not_at_all_for_a_display_change(
        qapp, qtbot, tmp_path, monkeypatch):
    """Finding 3: compared recordings' traces are kept for the life of the result, so an
    Update that moves no compute input runs the chain zero times."""
    session = Session()
    first = session.add_document(_doc(tmp_path, "A.dat", units="pA"))
    second = session.add_document(_doc(tmp_path, "B.dat", units="pA", sigma=9.0, seed=2))
    third = session.add_document(_doc(tmp_path, "C.dat", units="pA", sigma=6.0, seed=3))
    page, form = _page(qtbot, session, first, LOWPASS)
    calls = []
    real = psd_panel.conditioned_trace
    monkeypatch.setattr(psd_panel, "conditioned_trace",
                        lambda raw, fs, params: (calls.append(dict(params)),
                                                 real(raw, fs, params))[1])
    for index in range(page._compare_list.count()):
        page._compare_list.item(index).setCheckState(QtCore.Qt.Checked)
    page.set_signal_basis(PROCESSED)
    _update(page)
    assert calls == [LOWPASS, LOWPASS, LOWPASS], "once per recording, the same chain"
    assert page.curve_labels() == ["A.dat", "B.dat", "C.dat"]
    for curve, doc_id in zip(page.curves(), (first, second, third)):
        data, fs = _samples(session, doc_id)
        np.testing.assert_array_equal(
            curve.psd, _welch(page, np.asarray(real(data, fs, LOWPASS)), fs)[1])
    assert page._plot.titleLabel.text == (
        f"PSD — A.dat vs B.dat vs C.dat (processed: {LOWPASS_TEXT})")
    assert "applied to every curve" in page.details_text()
    assert _kept(page) == sorted([first, second, third])
    drawn = [curve.psd for curve in page.curves()]

    calls.clear()
    page._loglog_check.setChecked(not page._loglog_check.isChecked())      # a display toggle
    _update(page)
    page.set_view(VIEW_FFT)                                                # the view
    _update(page)
    page.set_view(VIEW_ALLPOINTS)
    _update(page)
    _update(page)                                                          # nothing at all
    page._start_spin.setValue(0.5)                                         # the window
    page._nperseg_combo.setCurrentText("1024")                             # a Welch setting
    _update(page)
    assert calls == [], "none of these is a new processed trace"
    assert not np.array_equal(page.curves()[0].psd, drawn[0])
    # A recording no longer compared lets its trace go; the others are read again.
    page._compare_list.item(1).setCheckState(QtCore.Qt.Unchecked)
    _update(page)
    assert calls == [] and _kept(page) == sorted([first, second])
    # Ticked again it is one pass, for that recording only; a new chain is one per curve.
    page._compare_list.item(1).setCheckState(QtCore.Qt.Checked)
    _update(page)
    assert calls == [LOWPASS]
    calls.clear()
    form["filter_cutoff_hz"] = 3000.0
    _update(page)
    assert len(calls) == 3 and _kept(page) == sorted([first, second, third])
    assert all(key[-1] == page._chain.key for key in page._processed_traces)


def test_the_kept_trace_is_never_reused_after_a_calibration_or_sample_rate_change(
        one, qtbot, monkeypatch):
    session, doc_id = one
    page, _form = _page(qtbot, session, doc_id, LOWPASS)
    page.set_signal_basis(PROCESSED)
    _update(page)
    before = page.psd_result()[1].copy()
    rates = []
    real = psd_panel.conditioned_trace
    monkeypatch.setattr(psd_panel, "conditioned_trace",
                        lambda raw, fs, params: (rates.append(float(fs)), real(raw, fs, params))[1])
    _update(page)
    assert rates == [], "an unchanged recording and chain read the kept trace"

    cal = session.calibration(doc_id, 0)
    session.set_calibration(doc_id, 0, replace(cal, gain=2.0))
    _update(page)
    assert rates == [50_000.0]
    data, fs = _samples(session, doc_id)
    np.testing.assert_array_equal(
        page.psd_result()[1], _welch(page, np.asarray(real(data, fs, LOWPASS)), fs)[1])
    assert not np.array_equal(page.psd_result()[1], before)
    assert len(page._processed_traces) == 1, "the trace of the old calibration is let go"

    rates.clear()
    session.set_sample_rate_declaration(
        doc_id, SampleRateDeclaration(mode="sample_rate", sample_rate_hz=100_000.0))
    _update(page)
    assert rates == [100_000.0]
    data, fs = _samples(session, doc_id)
    assert fs == 100_000.0
    i0, i1 = (int(round(spin.value() * fs)) for spin in (page._start_spin, page._end_spin))
    np.testing.assert_array_equal(
        page.psd_result()[1],
        _welch(page, np.asarray(real(data, fs, LOWPASS))[i0:i1], fs)[1])
    assert len(page._processed_traces) == 1


def test_the_kept_traces_go_when_a_recording_closes_or_the_page_leaves(qapp, qtbot, tmp_path):
    """Finding 10, at the page: the owner's two hooks (a closed recording, a page off screen)."""
    session = Session()
    first = session.add_document(_doc(tmp_path, "A.dat", units="pA"))
    second = session.add_document(_doc(tmp_path, "B.dat", units="pA", seed=2))
    page, _form = _page(qtbot, session, first, LOWPASS)
    page._compare_list.item(0).setCheckState(QtCore.Qt.Checked)
    page.set_signal_basis(PROCESSED)
    _update(page)
    assert _kept(page) == sorted([first, second])
    drawn = [curve.psd for curve in page.curves()]
    page.release_processed_traces(second)
    assert _kept(page) == [first]
    page.release_processed_traces()
    assert page._processed_traces == {}
    # The result on screen is untouched, and the next Update builds what it needs again.
    assert page.signal_basis() == PROCESSED
    for curve, psd in zip(page.curves(), drawn):
        np.testing.assert_array_equal(curve.psd, psd)
    _update(page)
    assert _kept(page) == sorted([first, second])


def test_busy_feedback_goes_through_the_safe_cursor_helper_and_the_status_line(
        one, qtbot, monkeypatch):
    """Finding 2: never QApplication.setOverrideCursor directly (it can crash on macOS)."""
    session, doc_id = one
    page, form = _page(qtbot, session, doc_id, LOWPASS)
    cursor, said = [], []
    monkeypatch.setattr(psd_panel, "begin_wait_cursor", lambda: (cursor.append("begin"), "token")[1])
    monkeypatch.setattr(psd_panel, "end_wait_cursor", lambda active: cursor.append(("end", active)))

    def forbidden(*_args):
        pytest.fail("the Noise page must not push a cursor itself")
    monkeypatch.setattr(QtWidgets.QApplication, "setOverrideCursor", forbidden)
    monkeypatch.setattr(QtWidgets.QApplication, "restoreOverrideCursor", forbidden)
    page.statusBar().messageChanged.connect(said.append)
    # A pass too short to notice says nothing and pushes nothing.
    page.set_signal_basis(PROCESSED)
    _update(page)
    assert said == [] and cursor == [("end", False)]
    # A long one names the recording on the status line before it starts, under the
    # helper's cursor, and leaves neither behind.
    cursor.clear()
    monkeypatch.setattr(psd_panel, "_BUSY_SAMPLES", 50_000)
    seen_while_running = []
    real = psd_panel.conditioned_trace
    monkeypatch.setattr(psd_panel, "conditioned_trace", lambda raw, fs, params: (
        seen_while_running.append(page.statusBar().currentMessage()), real(raw, fs, params))[1])
    form["filter_cutoff_hz"] = 3000.0
    _update(page)
    assert seen_while_running == ["Applying the Preprocess chain to A.dat…"]
    assert said == ["Applying the Preprocess chain to A.dat…", ""]
    assert cursor == ["begin", ("end", "token")]
    assert page.statusBar().currentMessage() == ""
    # A pass the kernels refuse gives the cursor back as well.
    cursor.clear()
    monkeypatch.setattr(psd_panel, "conditioned_trace",
                        lambda raw, fs, params: (_ for _ in ()).throw(ValueError("no")))
    form["filter_cutoff_hz"] = 3500.0
    assert page.inspector_adapter().update() is False
    assert cursor == ["begin", ("end", "token")]
    assert "setOverrideCursor" not in open(psd_panel.__file__, encoding="utf-8").read()


def test_every_baseline_advisory_becomes_a_note_of_its_recording(one, qtbot):
    """Finding 8: not only the degenerate-baseline warning; each cut to its first sentence."""
    session, doc_id = one
    chain = {**LOWPASS, **DRIFT}
    data, fs = _samples(session, doc_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with collect_baseline_warnings() as raised:
            conditioned_trace(data, fs, chain)
        page, _form = _page(qtbot, session, doc_id, chain)
        page.set_signal_basis(PROCESSED)
        _update(page)
    assert raised, "this chain is expected to raise an advisory on white noise"
    sentences = list(dict.fromkeys(psd_panel._first_sentence(w) for w in raised))
    assert any(str(w) != psd_panel._first_sentence(w) for w in raised), "something was cut"
    assert page.has_data() and page.curves()[0].note == " ".join(sentences)
    assert f"NOTE  {' '.join(sentences)}" in page.details_text().splitlines()
    assert psd_panel._first_sentence(
        "peak wander 3.2 pA in -> 4.1 pA out (+67%), measured on 0.125 s block medians. "
        "Wander with a period near the window is phase-inverted.") == (
        "peak wander 3.2 pA in -> 4.1 pA out (+67%), measured on 0.125 s block medians.")
    # The note is the curve's own: it is saved with it and shown on the other views.
    saved = capture_psd(page).payload
    assert saved["curves"][0]["note"] == " ".join(sentences)
    page.set_view(VIEW_FFT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _update(page)
    assert sentences[0] in page.caption_text(VIEW_FFT)


# -- saved state ----------------------------------------------------------------------------------

def _roundtrip(tmp_path, artifact):
    path = tmp_path / "signal.nanoscope"
    save_project(path, ProjectState(workspace=AnalyticalWorkspace(artifacts=[artifact])))
    return load_project(path).workspace.artifacts[0]


def test_the_choice_its_draft_and_the_result_survive_a_project_file(one, qtbot, tmp_path,
                                                                    monkeypatch):
    session, doc_id = one
    page, _form = _page(qtbot, session, doc_id, LOWPASS)
    page.set_signal_basis(PROCESSED)
    _update(page)
    drawn = page.psd_result()[1]
    record = decode_legacy(capture_psd(page), figure_id=str(uuid4()), name="Processed").record
    artifact = _roundtrip(tmp_path, encode_legacy(record.spec, record.resources))
    calls = []
    _count(monkeypatch, psd_panel.psd_mod, "welch_psd", calls)
    _count(monkeypatch, psd_panel, "conditioned_trace", calls)
    restored, _ = _page(qtbot, session, None, LOWPASS)
    report = restore_psd(restored, artifact)
    assert not report.skipped and not report.warnings and calls == []
    assert restored._signal_combo.currentData() == PROCESSED
    assert restored.signal_basis() == PROCESSED and restored.signal_chain() == LOWPASS_TEXT
    np.testing.assert_array_equal(restored.psd_result()[1], drawn)
    assert restored._plot.titleLabel.text == f"PSD — A.dat (processed: {LOWPASS_TEXT})"
    assert restored.details_text().startswith(f"Signal: processed — {LOWPASS_TEXT}")
    assert restored.inspector_adapter().pending().count == 0
    assert capture_psd(restored).payload["signal"] == artifact.payload["signal"]
    # The exact chain is restored with the result, so its export still carries it.
    assert json.loads(restored._plot_data()["signal_chain_params"].iloc[0]) == LOWPASS
    assert restored._processed_traces == {}, "a restore computes nothing and keeps no trace"

    # A draft of the choice alone is kept beside the inputs an older build reads.
    restored.set_signal_basis(RAW)
    native = capture_psd(restored)
    draft = native.payload["pending_analysis"]
    assert native.payload["config"]["signal_basis"] == PROCESSED
    assert "signal_basis" not in draft["inputs"] and draft["additive_inputs"] == {
        "signal_basis": RAW}
    # A Snapshot is the applied figure: processed, its chain named, and no draft.
    snapshot = capture_applied_record(decode_legacy(
        native, figure_id=str(uuid4()), name="Snapshot").record)
    frozen = encode_legacy(snapshot.spec, snapshot.resources).payload
    assert frozen["config"]["signal_basis"] == PROCESSED and "pending_analysis" not in frozen
    assert frozen["signal"] == artifact.payload["signal"]
    again, _ = _page(qtbot, session, None, LOWPASS)
    assert not restore_psd(again, _roundtrip(tmp_path, native)).skipped
    assert again._signal_combo.currentData() == RAW and again.signal_basis() == PROCESSED
    assert again.inspector_adapter().pending().reasons == ("signal basis",)
    assert calls == []


def test_a_restored_figure_seeds_only_the_view_that_was_on_screen_at_save(one, qtbot, tmp_path,
                                                                         monkeypatch):
    """Finding 1, at the page: the other two saved results may be of another signal, chain
    or span, so the Update that shows them computes them."""
    session, doc_id = one
    data, fs = _samples(session, doc_id)
    page, _form = _page(qtbot, session, doc_id, LOWPASS)
    page.set_signal_basis(PROCESSED)
    for view, _head in SIDE_VIEWS:
        page.set_view(view)
        _update(page)
    # Raw again, with the histogram in front: it alone is recomputed with the raw curves.
    page.set_signal_basis(RAW)
    _update(page)
    artifact = _roundtrip(tmp_path, capture_psd(page))
    assert "signal" not in artifact.payload and "signal_basis" not in artifact.payload["config"]
    assert artifact.payload["config"]["view"] == VIEW_ALLPOINTS

    restored, _ = _page(qtbot, session, None, LOWPASS)
    assert not restore_psd(restored, artifact).skipped
    assert list(restored._view_cache) == [VIEW_ALLPOINTS]
    raw = _side_view_kernels(restored, data, fs)
    calls = []
    _count(monkeypatch, psd_panel.ap_mod, "all_points_histogram", calls)
    _count(monkeypatch, psd_panel.psd_mod, "fft_magnitude", calls)
    _count(monkeypatch, psd_panel.psd_mod, "spectrogram_psd", calls)
    # The view on screen at save is drawn from its saved result: no kernel runs.
    _update(restored)
    assert calls == []
    np.testing.assert_array_equal(restored.all_points().counts, raw[VIEW_ALLPOINTS])
    # The other two are computed, from the raw trace, and say nothing of a chain.
    for view, head, kernel in ((VIEW_FFT, "FFT", "fft_magnitude"),
                               (VIEW_SPECTROGRAM, "Spectrogram", "spectrogram_psd")):
        calls.clear()
        restored.set_view(view)
        _update(restored)
        assert calls == [kernel]
        np.testing.assert_array_equal(_side_view_result(restored, view), raw[view])
        assert restored._view_titles[view] == f"{head} — A.dat"
        assert "Signal: processed" not in restored.caption_text(view)
        assert not BASIS_COLUMNS & set(restored._plot_data().columns)
    assert restored.signal_basis() == RAW

    # The older case, with no Signal choice in it: an FFT of an earlier time span.
    span, _ = _page(qtbot, session, doc_id, LOWPASS)
    span._start_spin.setValue(0.0)
    span._end_spin.setValue(0.5)
    span.set_view(VIEW_FFT)
    _update(span)
    span._end_spin.setValue(2.0)
    span.set_view(VIEW_PSD)
    _update(span)
    later, _ = _page(qtbot, session, None, LOWPASS)
    assert not restore_psd(later, _roundtrip(tmp_path, capture_psd(span))).skipped
    assert later._view_cache == {}
    later.set_view(VIEW_FFT)
    _update(later)
    np.testing.assert_array_equal(later.fft_spectrum().mag, raw[VIEW_FFT])


def test_a_figure_saved_without_the_field_opens_as_raw(one, qtbot, tmp_path):
    session, doc_id = one
    page, _form = _page(qtbot, session, doc_id, LOWPASS)
    raw_artifact = _roundtrip(tmp_path, capture_psd(page))
    assert "signal_basis" not in raw_artifact.payload["config"]
    # Onto a page that was showing Processed: the saved figure is Raw, so the page is.
    target, _ = _page(qtbot, session, doc_id, LOWPASS)
    target.set_signal_basis(PROCESSED)
    _update(target)
    report = restore_psd(target, raw_artifact)
    assert not report.skipped and not report.warnings
    assert target._signal_combo.currentData() == RAW and target.signal_basis() == RAW
    assert target._plot.titleLabel.text == "PSD — A.dat"
    assert target._processed_traces == {}
    np.testing.assert_array_equal(target.psd_result()[1], page.psd_result()[1])
    resaved = capture_psd(target).payload
    assert "signal" not in resaved and "signal_basis" not in resaved["config"]
    # A draft saved before the choice existed (no such input) is still a readable draft.
    page._nperseg_combo.setCurrentText("1024")
    old_draft = _roundtrip(tmp_path, capture_psd(page))
    assert set(old_draft.payload["pending_analysis"]) == {"version", "inputs"}
    assert "signal_basis" not in old_draft.payload["pending_analysis"]["inputs"]
    assert not restore_psd(target, old_draft).skipped
    assert target._nperseg_combo.currentText() == "1024"
    assert target._signal_combo.currentData() == RAW


def test_a_spectrum_an_older_build_recomputed_is_never_called_processed(one, qtbot, tmp_path):
    """A build without the choice keeps ``signal`` and ``signal_basis`` when it saves, but a
    spectrum it recomputes is of the raw trace. The record's fingerprint tells them apart."""
    session, doc_id = one
    page, _form = _page(qtbot, session, doc_id, LOWPASS)
    raw = capture_psd(page)
    page.set_signal_basis(PROCESSED)
    _update(page)
    processed = capture_psd(page)
    # Saved again untouched (what an older build does without an Update): still processed.
    untouched = deepcopy(processed.payload)
    kept, _ = _page(qtbot, session, None, LOWPASS)
    assert not restore_psd(kept, replace(processed, payload=untouched)).warnings
    assert kept.signal_basis() == PROCESSED
    # Recomputed there: its raw curves under this build's carried-forward fields.
    payload = deepcopy(raw.payload)
    payload["config"]["signal_basis"] = PROCESSED
    payload["signal"] = deepcopy(processed.payload["signal"])
    stale, _ = _page(qtbot, session, None, LOWPASS)
    report = restore_psd(stale, replace(raw, payload=payload))
    assert not report.skipped
    assert report.warnings == ("the saved Noise spectrum is of the raw signal; Signal is set "
                               "to match it",)
    assert stale.signal_basis() == RAW and stale._signal_combo.currentData() == RAW
    assert stale._plot.titleLabel.text == "PSD — A.dat"
    assert not BASIS_COLUMNS & set(stale._plot_data().columns)
    np.testing.assert_array_equal(stale.psd_result()[1], np.asarray(raw.tables[
        raw.payload["curves"][0]["table"]]["psd"]))
    resaved = capture_psd(stale).payload
    assert "signal" not in resaved and "signal_basis" not in resaved["config"]
    # A capture that no longer names the choice does not inherit the old record's.
    old = decode_legacy(processed, figure_id=str(uuid4()), name="was processed").record
    merged = merge_native_capture(capture_psd(stale), old)
    merged = encode_legacy(merged.spec, merged.resources)
    assert "signal" not in merged.payload and "signal_basis" not in merged.payload["config"]


def test_a_template_carries_the_choice_and_one_without_it_keeps_the_pages(one, qtbot):
    session, doc_id = one
    page, _form = _page(qtbot, session, doc_id, LOWPASS)
    old_settings, appearance = page.template_settings()      # as saved before the choice
    assert "signal_basis" not in old_settings
    page.set_signal_basis(PROCESSED)
    _update(page)
    settings, _ = page.template_settings()
    assert settings["signal_basis"] == PROCESSED
    other, _ = _page(qtbot, session, doc_id, LOWPASS)
    other.apply_template_settings(settings, appearance)
    assert other._signal_combo.currentData() == PROCESSED and other.signal_basis() == RAW
    assert other.inspector_adapter().pending().reasons == ("signal basis",)
    _update(other)
    np.testing.assert_array_equal(other.psd_result()[1], page.psd_result()[1])
    # Finding 9: a template without the key (every one saved before the choice) leaves the
    # page's choice alone, like any other key a template does not carry.
    other.apply_template_settings({**old_settings, "nperseg": "1024"}, appearance)
    assert other._signal_combo.currentData() == PROCESSED
    assert other.inspector_adapter().pending().reasons == ("nperseg",)
    _update(other)
    assert other.signal_basis() == PROCESSED
    # One that names Raw sets it.
    other.apply_template_settings({**old_settings, "signal_basis": RAW}, appearance)
    assert other._signal_combo.currentData() == RAW
    with pytest.raises(ValueError):
        other.apply_template_settings({**settings, "signal_basis": "ml"}, appearance)
    # Standing alone (no Update button) a template acts at once, like the control.
    alone, _ = _page(qtbot, session, doc_id, LOWPASS, managed=False)
    alone.apply_template_settings(settings, appearance)
    assert alone.signal_basis() == PROCESSED
    np.testing.assert_array_equal(alone.psd_result()[1], page.psd_result()[1])
    alone.apply_template_settings(old_settings, appearance)
    assert alone.signal_basis() == PROCESSED


# -- in the app: the Detection form is the chain, the figure record follows --------------------------

def _app_with_noise(qtbot, tmp_path, monkeypatch, name="noise.csv"):
    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    recording = tmp_path / name
    _noisy_csv(recording)
    window = _main_window(qtbot)
    with qtbot.waitSignal(window.projectRestored, timeout=20_000):
        window._begin_restore(ProjectState(documents=[DocumentRef(str(recording))]))
    window._analysis_panel.detection_panel.set_params({"filter_cutoff_hz": 2000.0})
    assert window._analysis_panel.detection_params()["filter_cutoff_hz"] == 2000.0
    window._open_psd()
    page = window._psd_window
    H.press_update(window._analysis_explorer)
    assert page.has_data() and page.signal_basis() == RAW
    return window, page


def _install(window, doc):
    """Open a synthetic recording in ``window`` and select it."""
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    return doc_id


def _status(window, page):
    owner = window._figure_workspace
    key = owner.id_for_page(page)
    return owner.get(key).spec.result.status, window._analysis_explorer_status(key, page)[0]


def test_in_the_app_a_chain_edit_turns_the_processed_figure_stale_until_update(
        qapp, qtbot, tmp_path, monkeypatch):
    window, page = _app_with_noise(qtbot, tmp_path, monkeypatch)
    explorer, owner = window._analysis_explorer, window._figure_workspace
    key = owner.id_for_page(page)
    raw_psd = page.psd_result()[1]
    detection = window._analysis_panel.detection_panel
    # A raw figure is current whatever the form does.
    detection.set_params({"filter_cutoff_hz": 3000.0})
    assert _status(window, page) == ("current", "CURRENT")
    detection.set_params({"filter_cutoff_hz": 2000.0})

    explorer.reveal("sources")
    assert explorer.inspector().pane(key).data_pane.group("sources").body.isAncestorOf(
        page._signal_combo)
    _pick(page._signal_combo, data=PROCESSED)
    assert explorer.adapter(key).pending().reasons == ("signal basis",)
    assert _status(window, page) == ("stale", "STALE")
    np.testing.assert_array_equal(page.psd_result()[1], raw_psd)
    H.press_update(explorer)
    assert _status(window, page) == ("current", "CURRENT")
    assert page.signal_basis() == PROCESSED and page.signal_chain() == LOWPASS_TEXT
    chan = window._session.calibrated(page.curves()[0].doc_id, page.curves()[0].channel)
    data, fs = np.asarray(chan.data), float(chan.sample_rate_hz)
    chain = window._analysis_panel.detection_params()
    np.testing.assert_array_equal(
        page.psd_result()[1], _welch(page, np.asarray(conditioned_trace(data, fs, chain)), fs)[1])
    drawn = page.psd_result()[1]

    # The form moves: the figure needs an update and says so; nothing is recomputed.
    detection.set_params({"filter_cutoff_hz": 5000.0})
    assert explorer.adapter(key).pending().reasons == ("Preprocess chain changed",)
    assert _status(window, page) == ("stale", "STALE")
    assert "Preprocess chain changed" in explorer.toolbar().update_button.toolTip()
    np.testing.assert_array_equal(page.psd_result()[1], drawn)
    # Put back, it is current again without a computation.
    detection.set_params({"filter_cutoff_hz": 2000.0})
    assert explorer.adapter(key).pending().count == 0
    assert _status(window, page) == ("current", "CURRENT")
    assert "Preprocess chain changed" not in explorer.toolbar().update_button.toolTip()
    # Normalize is left out of the signal, so turning it on is no change either.
    detection.set_params({"normalize_enabled": True})
    assert window._analysis_panel.detection_params()["normalize_enabled"] is True
    assert explorer.adapter(key).pending().count == 0
    assert _status(window, page) == ("current", "CURRENT")
    detection.set_params({"normalize_enabled": False})
    detection.set_params({"filter_cutoff_hz": 5000.0})
    H.press_update(explorer)
    assert _status(window, page) == ("current", "CURRENT")
    assert page.signal_chain() == "Low-pass Bessel 8 · 5 kHz"
    assert not np.array_equal(page.psd_result()[1], drawn)
    drawn = page.psd_result()[1]

    # Figure Undo takes back a look, not the choice.
    _edit_double_click(page._plot, page._plot.titleLabel, "Filtered noise")
    assert page.appearance_state().get("title") == "Filtered noise"
    H.press_undo(explorer)
    assert "title" not in page.appearance_state()
    assert page._signal_combo.currentData() == PROCESSED and page.signal_basis() == PROCESSED
    assert page._plot.titleLabel.text.endswith("(processed: Low-pass Bessel 8 · 5 kHz)")
    np.testing.assert_array_equal(page.psd_result()[1], drawn)

    # Duplicate: the copy is the same processed figure.
    copy_key = owner.duplicate(key)
    copy = owner.page(copy_key)
    assert copy is not page and copy._signal_combo.currentData() == PROCESSED
    assert copy.signal_basis() == PROCESSED
    assert copy.signal_chain() == "Low-pass Bessel 8 · 5 kHz"
    np.testing.assert_array_equal(copy.psd_result()[1], drawn)
    assert explorer.adapter(copy_key).pending().count == 0
    # Both follow the one form.
    detection.set_params({"filter_cutoff_hz": 2000.0})
    for figure in (key, copy_key):
        assert owner.get(figure).spec.result.status == "stale"
    detection.set_params({"filter_cutoff_hz": 5000.0})
    for figure in (key, copy_key):
        assert owner.get(figure).spec.result.status == "current"
    # A figure that is not resident is judged from what its record says it was made with.
    assert owner.suspend(key) and owner.page(key) is None
    detection.set_params({"filter_cutoff_hz": 2000.0})
    for figure in (key, copy_key):
        assert owner.get(figure).spec.result.status == "stale"
    detection.set_params({"normalize_enabled": True})          # no part of the judgement
    for figure in (key, copy_key):
        assert owner.get(figure).spec.result.status == "stale"
    detection.set_params({"filter_cutoff_hz": 5000.0})
    for figure in (key, copy_key):
        assert owner.get(figure).spec.result.status == "current"
    detection.set_params({"normalize_enabled": False})
    # Opened again while the form differs, its own page takes the judgement over.
    detection.set_params({"filter_cutoff_hz": 2000.0})
    assert owner.activate(key)
    woken = owner.page(key)
    assert woken.signal_basis() == PROCESSED
    assert explorer.adapter(key).pending().reasons == ("Preprocess chain changed",)
    assert _status(window, woken) == ("stale", "STALE")
    np.testing.assert_array_equal(woken.psd_result()[1], drawn)

    # A recipe made from the figure starts a new one on the same choice, not yet computed.
    settings = woken.template_settings()[0]
    assert settings["signal_basis"] == PROCESSED
    fresh_key = owner.create("noise", initial_recipe={"window_template": settings})
    fresh = owner.page(fresh_key)
    assert fresh._signal_combo.currentData() == PROCESSED and not fresh.has_data()
    H.press_update(explorer)
    assert fresh.has_data() and fresh.signal_basis() == PROCESSED
    assert fresh.signal_chain() == LOWPASS_TEXT
    window._dirty = False


def test_in_the_app_a_project_save_and_reopen_restores_the_choice_and_the_spectrum(
        qapp, qtbot, tmp_path, monkeypatch):
    window, page = _app_with_noise(qtbot, tmp_path, monkeypatch)
    explorer = window._analysis_explorer
    page.set_signal_basis(PROCESSED)
    H.press_update(explorer)
    assert page.signal_basis() == PROCESSED
    drawn = page.psd_result()
    project = tmp_path / "processed-noise.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_psd()
    again = reopened._psd_window
    assert again._signal_combo.currentData() == PROCESSED
    assert again.signal_basis() == PROCESSED and again.signal_chain() == LOWPASS_TEXT
    for actual, expected in zip(again.psd_result(), drawn):
        np.testing.assert_array_equal(actual, expected)
    assert again._plot.titleLabel.text.endswith(f"(processed: {LOWPASS_TEXT})")
    assert again.details_text().startswith(f"Signal: processed — {LOWPASS_TEXT}")
    assert reopened._analysis_explorer.adapter(
        reopened._figure_workspace.id_for_page(again)).pending().count == 0
    assert _status(reopened, again) == ("frozen", "SAVED RESULT")
    assert not reopened._dirty, "looking at a reopened figure is not a change"
    # The reopened form is the saved one, and Update computes the same processed spectrum.
    H.press_update(reopened._analysis_explorer)
    assert again.signal_basis() == PROCESSED
    np.testing.assert_array_equal(again.psd_result()[1], drawn[1])
    reopened._dirty = False


def test_in_the_app_a_restored_raw_figure_never_serves_a_processed_view_as_raw(
        qapp, qtbot, tmp_path, monkeypatch):
    """Finding 1: FFT, spectrogram and histogram looked at on Processed, then Raw / PSD and
    Update, then a restore three ways. Each side view after Update is the raw one, titled
    and exported as raw."""
    window, page = _app_with_noise(qtbot, tmp_path, monkeypatch)
    explorer, owner = window._analysis_explorer, window._figure_workspace
    key = owner.id_for_page(page)
    page.set_signal_basis(PROCESSED)
    for view, head in SIDE_VIEWS:
        page.set_view(view)
        H.press_update(explorer)
        assert page._view_titles[view] == f"{head} — noise.csv (processed: {LOWPASS_TEXT})"
    page.set_signal_basis(RAW)
    page.set_view(VIEW_PSD)
    H.press_update(explorer)
    assert page.signal_basis() == RAW and _status(window, page) == ("current", "CURRENT")
    chan = window._session.calibrated(page.curves()[0].doc_id, page.curves()[0].channel)
    data, fs = np.asarray(chan.data), float(chan.sample_rate_hz)
    raw = _side_view_kernels(page, data, fs)
    processed = _side_view_kernels(
        page, np.asarray(conditioned_trace(data, fs, window._analysis_panel.detection_params())), fs)
    # What the page still holds from its Processed visits is what a restore is handed.
    for view, _head in SIDE_VIEWS:
        np.testing.assert_array_equal(_side_view_result(page, view), processed[view])
        assert not np.array_equal(processed[view], raw[view])

    def side_views_are_raw(shell, figure):
        plots = {VIEW_FFT: figure._fft_plot, VIEW_SPECTROGRAM: figure._spec_plot,
                 VIEW_ALLPOINTS: figure._hist_plot}
        for view, head in SIDE_VIEWS:
            figure.set_view(view)
            H.press_update(shell)
            assert figure.current_view() == view
            np.testing.assert_array_equal(_side_view_result(figure, view), raw[view])
            assert figure._view_titles[view] == f"{head} — noise.csv"
            assert plots[view].titleLabel.text == f"{head} — noise.csv"
            assert "Signal: processed" not in figure.caption_text(view)
            assert not BASIS_COLUMNS & set(figure._plot_data().columns)
            assert figure.signal_basis() == RAW and "Signal:" not in figure.details_text()

    # 1. Project save, for the reopen below.
    project = tmp_path / "raw-after-processed.nanoscope"
    assert window._write_project(str(project))
    # 2. Duplicate.
    copy_key = owner.duplicate(key)
    copy = owner.page(copy_key)
    assert copy is not page and copy.signal_basis() == RAW
    side_views_are_raw(explorer, copy)
    # 3. Suspend and resume of the original.
    assert owner.suspend(key) and owner.page(key) is None
    assert owner.activate(key)
    woken = owner.page(key)
    assert woken is not page and woken.signal_basis() == RAW
    side_views_are_raw(explorer, woken)
    window._dirty = False
    # 1 again: the saved project, reopened.
    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    reopened._open_psd()
    again = reopened._psd_window
    assert again.signal_basis() == RAW and again._signal_combo.currentData() == RAW
    side_views_are_raw(reopened._analysis_explorer, again)
    reopened._dirty = False


def test_in_the_app_looking_at_another_table_is_a_judgement_and_not_an_unsaved_change(
        qapp, qtbot, tmp_path, monkeypatch):
    """Finding 5: each event table has its own form, so selecting another recording can move
    the chain under a processed figure. The figure says so; the project is not marked."""
    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    window = _main_window(qtbot)
    detection = window._analysis_panel.detection_panel
    doc_a = _install(window, _doc(tmp_path, "A.dat", units="pA"))
    detection.set_params({"filter_cutoff_hz": 2000.0})           # A's form
    doc_b = _install(window, _doc(tmp_path, "B.dat", units="pA", seed=5))
    detection.set_params({"filter_cutoff_hz": 5000.0})           # B's form
    window._on_selection(doc_a, 0)
    assert window._analysis_panel.detection_params()["filter_cutoff_hz"] == 2000.0
    window._open_psd()
    page = window._psd_window
    explorer, owner = window._analysis_explorer, window._figure_workspace
    key = owner.id_for_page(page)
    H.press_update(explorer)
    page.set_signal_basis(PROCESSED)
    H.press_update(explorer)
    assert page.signal_basis() == PROCESSED and _status(window, page) == ("current", "CURRENT")
    drawn = page.psd_result()[1]
    edits = []
    page.pendingChanged.connect(lambda: edits.append(1))

    window._dirty = False
    window._on_selection(doc_b, 0)
    qtbot.wait(20)
    assert window._analysis_panel.detection_params()["filter_cutoff_hz"] == 5000.0
    assert explorer.adapter(key).pending().reasons == ("Preprocess chain changed",)
    assert _status(window, page) == ("stale", "STALE")
    assert "Preprocess chain changed" in explorer.toolbar().update_button.toolTip()
    assert not window._dirty and edits == []
    window._on_selection(doc_a, 0)
    qtbot.wait(20)
    assert explorer.adapter(key).pending().count == 0
    assert _status(window, page) == ("current", "CURRENT")
    assert "Preprocess chain changed" not in explorer.toolbar().update_button.toolTip()
    assert not window._dirty and edits == []
    np.testing.assert_array_equal(page.psd_result()[1], drawn)
    # A real edit of the figure still arms the save prompt.
    page._start_spin.setValue(0.5)
    assert window._dirty and edits
    window._dirty = False


def test_in_the_app_opening_a_reopened_figure_whose_chain_moved_is_not_an_unsaved_change(
        qapp, qtbot, tmp_path, monkeypatch):
    window, page = _app_with_noise(qtbot, tmp_path, monkeypatch)
    page.set_signal_basis(PROCESSED)
    H.press_update(window._analysis_explorer)
    window._analysis_panel.detection_panel.set_params({"filter_cutoff_hz": 3000.0})
    assert _status(window, page) == ("stale", "STALE")
    project = tmp_path / "chain-moved.nanoscope"
    assert window._write_project(str(project))
    window._dirty = False

    reopened = _main_window(qtbot)
    with qtbot.waitSignal(reopened.projectRestored, timeout=20_000):
        reopened._open_project(str(project))
    qtbot.wait(50)
    assert not reopened._dirty
    reopened._open_psd()
    qtbot.wait(50)
    again = reopened._psd_window
    key = reopened._figure_workspace.id_for_page(again)
    assert again.signal_basis() == PROCESSED and again.signal_chain() == LOWPASS_TEXT
    assert reopened._analysis_panel.detection_params()["filter_cutoff_hz"] == 3000.0
    assert reopened._analysis_explorer.adapter(key).pending().reasons == (
        "Preprocess chain changed",)
    assert reopened._figure_workspace.get(key).spec.result.status == "stale"
    assert not reopened._dirty, "the judgement is not an edit of the figure"
    reopened._dirty = False


def test_in_the_app_a_typed_title_keeps_the_marker_in_the_figure_and_in_a_bundle(
        qapp, qtbot, tmp_path, monkeypatch):
    """Finding 4, through the figure's own title editor and what then leaves the app."""
    window, page = _app_with_noise(qtbot, tmp_path, monkeypatch)
    explorer, owner = window._analysis_explorer, window._figure_workspace
    key = owner.id_for_page(page)
    _edit_double_click(page._plot, page._plot.titleLabel, "Pore 7")
    assert page._plot.titleLabel.text == "Pore 7", "a raw result's typed title is as typed"
    page.set_signal_basis(PROCESSED)
    H.press_update(explorer)
    marked = f"Pore 7 (processed: {LOWPASS_TEXT})"
    assert page.appearance_state().get("title") == "Pore 7", "the typed words stay the user's"
    assert page._plot.titleLabel.text == marked
    bundle = window._export_analysis_bundle(key, str(tmp_path / "typed.zip"))
    assert bundle is not None
    with zipfile.ZipFile(bundle) as archive:
        svg = archive.read("figure.svg").decode("utf-8")
        table = pd.read_csv(archive.open("data.csv"))
    assert "Pore 7 (processed: Low-pass Bessel 8" in svg
    assert set(table["signal_basis"]) == {PROCESSED}
    used = psd_panel._noise_chain_params(window._analysis_panel.detection_params())
    assert not any(name.startswith("normalize") for name in used) and used["filter_cutoff_hz"] == 2000.0
    assert [json.loads(text) for text in set(table["signal_chain_params"])] == [
        json.loads(json.dumps(used, default=psd_panel._plain_json))]
    # The editor opens on the words that were typed, and what it keeps is marked once.
    _edit_double_click(page._plot, page._plot.titleLabel, "Pore 8")
    assert page.appearance_state().get("title") == "Pore 8"
    assert page._plot.titleLabel.text == f"Pore 8 (processed: {LOWPASS_TEXT})"
    # Raw again is the typed title alone.
    page.set_signal_basis(RAW)
    H.press_update(explorer)
    assert page._plot.titleLabel.text == "Pore 8"
    window._dirty = False


def test_in_the_app_a_closed_recording_and_a_suspended_page_keep_no_processed_trace(
        qapp, qtbot, tmp_path, monkeypatch):
    """Finding 10: the owner tells the page when a recording closes, and when it leaves."""
    monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
    window = _main_window(qtbot)
    doc_a = _install(window, _doc(tmp_path, "A.dat", units="pA"))
    doc_b = _install(window, _doc(tmp_path, "B.dat", units="pA", seed=5))
    window._on_selection(doc_a, 0)
    window._analysis_panel.detection_panel.set_params({"filter_cutoff_hz": 2000.0})
    window._open_psd()
    page = window._psd_window
    explorer, owner = window._analysis_explorer, window._figure_workspace
    key = owner.id_for_page(page)
    H.press_update(explorer)
    page._compare_list.item(0).setCheckState(QtCore.Qt.Checked)
    page.set_signal_basis(PROCESSED)
    H.press_update(explorer)
    assert page.curve_labels() == ["A.dat", "B.dat"] and _kept(page) == sorted([doc_a, doc_b])

    window._session.close_document(doc_b)
    qtbot.wait(30)
    assert _kept(page) == [doc_a]
    assert owner.suspend(key) and owner.page(key) is None
    assert page._processed_traces == {}
    window._dirty = False
