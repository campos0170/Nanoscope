"""Noise: Export data writes the spectrum's real values, and a closed recording blocks nothing.

N1. With the log-log box ticked (the default) Noise ▸ Export data wrote log10 numbers under
``frequency_hz``, ``psd_value`` and ``peak_amplitude``: the export read ``item.getData()``,
which pyqtgraph answers in DISPLAY coordinates. The file must hold the computed values
whatever the box says, for every view the export covers.

N2. Closing the recording a Noise figure is pinned to made Save Project fail ("A figure could
not be checkpointed"), and with that figure in front neither switching to another figure nor
Delete worked: the page read the closed recording's document out of its last gather. The
figure keeps its saved spectrum, says its recording is not open (as Event waveforms does), and
is saved, switched away from and deleted like any other.

N2, after the review. (a) The figure's Start, End and RMS band belong to the recording they
were set for, and that recording is its persistent key, not the per-load ``doc_id``: after the
recording is closed and opened again (or after a project is reopened without it and the
recording is opened afterwards) Update must keep the four values, and only a recording that
really is another one (another key, another length or another rate) gets the defaults, also
when the page lists the new load before Update (another recording closing, a draft being
checkpointed) and, on the project route, when the file ends before the stretch the saved
spectrum used. (b) The Source box shows a closed recording as the one disabled "Unavailable
recording" row, offers it again as soon as it opens, and no row for a closed recording can zero
the four values. (c) A comparison recording closed before Save is still a comparison after the
project is saved and reopened twice.

Synthetic data only; offscreen.
"""
from __future__ import annotations

import os
import zipfile
from dataclasses import replace

import nanoscope  # noqa: F401  -- Qt bootstrap before the panel imports
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore

from nanoscope.core.analysis_workspace import capture_psd, restore_psd
from nanoscope.core.project import AnalyticalWorkspace, ProjectState, load_project, save_project
from nanoscope.core.session import Session
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.psd_panel import (VIEW_ALLPOINTS, VIEW_FFT, VIEW_PSD, VIEW_SPECTROGRAM,
                                     PSDWindow)

from . import p1_hosting as H
from .test_plot_window_routes import _main_window
from .test_s3_units_psd_window import _doc


# -- N1: the file holds the real values ------------------------------------------------------------

def _window(qtbot, tmp_path, *, compared=False):
    """A Noise window on one white-noise recording (and, with ``compared``, a second one)."""
    session = Session()
    first = session.add_document(_doc(tmp_path, "A.dat", units="pA"))
    if compared:
        session.add_document(_doc(tmp_path, "B.dat", units="pA", sigma=9.0, seed=2))
    window = PSDWindow(session)
    qtbot.addWidget(window)
    window.set_target(first, 0)
    window.show()
    return window


def _export(window, tmp_path, name):
    out = window.export_data(str(tmp_path / name))
    assert out is not None, "the export was refused"
    return pd.read_csv(out, float_precision="round_trip", low_memory=False)


def _part(frame, part):
    return frame[frame["part"] == part]


def _same_numbers(left, right, *columns):
    for column in columns:
        np.testing.assert_array_equal(left[column].to_numpy(), right[column].to_numpy(),
                                      err_msg=f"{column} depends on the log-log box")


@pytest.mark.parametrize("asd", [False, True], ids=["PSD", "ASD"])
def test_psd_export_is_the_computed_spectrum_whatever_the_log_log_box_says(
        qapp, qtbot, tmp_path, asd):
    window = _window(qtbot, tmp_path)
    freqs, psd = window.psd_result()
    window.set_asd(asd)
    assert window._loglog_check.isChecked(), "log-log is the default"
    ticked = _export(window, tmp_path, "ticked.csv")
    window.set_log_log(False)
    unticked = _export(window, tmp_path, "unticked.csv")

    shown = np.sqrt(psd[1:]) if asd else psd[1:]
    for frame in (ticked, unticked):
        curve = _part(frame, "curve")
        np.testing.assert_array_equal(curve["frequency_hz"].to_numpy(), freqs[1:])
        np.testing.assert_array_equal(curve["display_value"].to_numpy(), shown)
        np.testing.assert_allclose(curve["psd_value"].to_numpy(), psd[1:], rtol=1e-12)
        assert curve["frequency_hz"].max() == pytest.approx(freqs[-1]), "a log10 would be < 5"
    _same_numbers(ticked, unticked, "part", "point_index", "frequency_hz", "display_value",
                  "psd_value")
    x, y = window.plotted_data()
    np.testing.assert_array_equal(_part(ticked, "curve")["frequency_hz"].to_numpy(), x)
    np.testing.assert_array_equal(_part(ticked, "curve")["display_value"].to_numpy(), y)


@pytest.mark.parametrize("asd", [False, True], ids=["PSD", "ASD"])
def test_fit_line_export_is_the_fitted_power_law_whatever_the_log_log_box_says(
        qapp, qtbot, tmp_path, asd):
    window = _window(qtbot, tmp_path)
    window.set_asd(asd)
    fit = window.summary()["flicker"]
    low, high = window.summary()["flicker_band"]
    ticked = _export(window, tmp_path, "ticked.csv")
    window.set_log_log(False)
    unticked = _export(window, tmp_path, "unticked.csv")

    _same_numbers(_part(ticked, "flicker_fit"), _part(unticked, "flicker_fit"),
                  "frequency_hz", "display_value", "psd_value")
    for frame in (ticked, unticked):
        line = _part(frame, "flicker_fit")
        f = line["frequency_hz"].to_numpy()
        assert len(f) == 60 and f[0] == pytest.approx(low) and f[-1] == pytest.approx(high)
        power = fit["A1Hz"] * f ** -fit["alpha"]
        np.testing.assert_allclose(line["psd_value"].to_numpy(), power, rtol=1e-9)
        np.testing.assert_allclose(line["display_value"].to_numpy(),
                                   np.sqrt(power) if asd else power, rtol=1e-9)


def test_fft_export_is_the_computed_spectrum_whatever_the_log_log_box_says(
        qapp, qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    window.set_view(VIEW_FFT)
    spec = window.fft_spectrum()
    assert window._loglog_check.isChecked()
    ticked = _export(window, tmp_path, "ticked.csv")
    window.set_log_log(False)
    unticked = _export(window, tmp_path, "unticked.csv")

    for frame in (ticked, unticked):
        curve = _part(frame, "fft_curve")
        np.testing.assert_array_equal(curve["frequency_hz"].to_numpy(), spec.freqs)
        np.testing.assert_array_equal(curve["peak_amplitude"].to_numpy(), spec.mag)
    _same_numbers(ticked, unticked, "frequency_hz", "peak_amplitude")


def test_spectrogram_export_is_the_computed_grid_whatever_the_boxes_say(qapp, qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    window.set_view(VIEW_SPECTROGRAM)
    sg = window.spectrogram()
    n_f, n_t = sg.psd.shape
    frames = []
    for log_log in (True, False):
        for log_frequency in (False, True):
            window.set_log_log(log_log)
            window._spec_logf_check.setChecked(log_frequency)
            frames.append(_part(_export(window, tmp_path, f"{log_log}-{log_frequency}.csv"),
                                "spectrogram_cell"))
    for cells in frames:
        np.testing.assert_array_equal(cells["psd_value"].to_numpy(), sg.psd.T.reshape(-1))
        np.testing.assert_array_equal(cells["frequency_hz"].to_numpy(), np.tile(sg.freqs, n_t))
        # The one column that IS a logarithm says so in its name.
        np.testing.assert_allclose(cells["display_log10_psd"].to_numpy(),
                                   np.log10(sg.psd.T.reshape(-1)), rtol=1e-5)
    for cells in frames[1:]:
        _same_numbers(frames[0], cells, "psd_value", "frequency_hz", "display_log10_psd")


def test_all_points_export_is_the_computed_histogram_whatever_the_log_log_box_says(
        qapp, qtbot, tmp_path):
    window = _window(qtbot, tmp_path)
    window.set_view(VIEW_ALLPOINTS)
    hist = window.all_points()
    frames = []
    for log_log in (True, False):
        window.set_log_log(log_log)
        frames.append(_part(_export(window, tmp_path, f"{log_log}.csv"), "histogram_bin"))
    for bins in frames:
        np.testing.assert_array_equal(bins["bin_left"].to_numpy(), hist.edges[:-1])
        np.testing.assert_array_equal(bins["bin_right"].to_numpy(), hist.edges[1:])
        np.testing.assert_array_equal(bins["bin_center"].to_numpy(), hist.centers)
        np.testing.assert_array_equal(bins["count"].to_numpy(), hist.counts)
    _same_numbers(frames[0], frames[1], "bin_left", "bin_right", "bin_center", "count")


def test_every_overlaid_curve_exports_its_own_computed_spectrum(qapp, qtbot, tmp_path):
    window = _window(qtbot, tmp_path, compared=True)
    window.set_compare(window.comparable_labels())
    curves = window.curves()
    assert len(curves) == 2
    for log_log in (True, False):
        window.set_log_log(log_log)
        rows = _part(_export(window, tmp_path, f"{log_log}.csv"), "curve")
        for curve in curves:
            mine = rows[rows["source_key"] == curve.key]
            np.testing.assert_array_equal(mine["frequency_hz"].to_numpy(), curve.freqs[1:])
            np.testing.assert_array_equal(mine["psd_value"].to_numpy(), curve.psd[1:])


# -- N2: a figure whose recording closed ------------------------------------------------------------

def _install(window, doc):
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    return doc_id


class _Rig:
    """The app with a Noise figure computed on recording A, pinned to it, and an I-V figure."""

    def __init__(self, qtbot, tmp_path, monkeypatch):
        monkeypatch.setattr(MainWindow, "_offer_startup_recovery", lambda self: None)
        self.tmp_path = tmp_path
        self.window = window = _main_window(qtbot)
        self.first = _install(window, _doc(tmp_path, "A.dat", units="pA"))
        self.second = _install(window, _doc(tmp_path, "B.dat", units="pA", seed=5))
        window._on_selection(self.first, 0)
        window._open_psd()
        self.explorer, self.owner = window._analysis_explorer, window._figure_workspace
        self.key = self.owner.current_id()
        H.press_update(self.explorer)
        self.page = self.owner.page(self.key)
        assert self.page.has_data() and self.page.curves()[0].doc_id == self.first
        self.spectrum = self.page.psd_result()[1].copy()
        self.other = self.owner.create("iv")
        self.owner.activate(self.key)
        assert self.owner.current_id() == self.key
        self.failures = []
        self.owner.requestFailed.connect(lambda _key, reason: self.failures.append(reason))

    def close_recording(self):
        """File ▸ Close document on the figure's recording."""
        self.window._close_document(self.first)
        assert not self.window._session.has_document(self.first)

    def saved_spectrum(self, page=None):
        return (page or self.page).psd_result()[1]

    def apply(self, *values):
        """Type Start, End and the RMS band, then press Update plot: they are the figure's own."""
        page = self.page
        for spin, value in zip((page._start_spin, page._end_spin, page._rms_lo_spin,
                                page._rms_hi_spin), values):
            spin.setValue(value)
        H.press_update(self.explorer)
        assert _four(page) == values and page.inspector_adapter().pending().count == 0


@pytest.fixture
def rig(qapp, qtbot, tmp_path, monkeypatch):
    rig = _Rig(qtbot, tmp_path, monkeypatch)
    yield rig
    rig.window._dirty = False       # the test's own edits are not the quit guard's business


def test_a_noise_figure_bundle_holds_the_real_values_too(rig, tmp_path):
    """The bundle's data.csv is the same table the Export data command writes (N1)."""
    freqs, psd = rig.page.psd_result()
    assert rig.page._loglog_check.isChecked()
    bundle = rig.window._export_analysis_bundle(rig.key, str(tmp_path / "noise.zip"))
    assert bundle is not None
    with zipfile.ZipFile(bundle) as archive:
        table = pd.read_csv(archive.open("data.csv"), float_precision="round_trip",
                            low_memory=False)
    curve = table[table["part"] == "curve"]
    np.testing.assert_array_equal(curve["frequency_hz"].to_numpy(), freqs[1:])
    np.testing.assert_array_equal(curve["psd_value"].to_numpy(), psd[1:])


def test_closing_the_figures_recording_still_lets_the_project_be_saved(rig, blocked_modals):
    rig.close_recording()
    path = rig.tmp_path / "closed.nanoscope"
    assert rig.window._write_project(str(path)) is True, rig.failures
    assert not blocked_modals, blocked_modals
    assert path.exists() and load_project(path) is not None


def test_closing_the_figures_recording_still_lets_the_user_switch_to_another_figure(rig):
    rig.close_recording()
    assert rig.owner.activate(rig.other) is True, rig.failures
    assert rig.owner.current_id() == rig.other
    assert rig.owner.activate(rig.key) is True, rig.failures
    assert rig.owner.current_id() == rig.key
    np.testing.assert_array_equal(rig.saved_spectrum(rig.owner.page(rig.key)), rig.spectrum)


def test_closing_the_figures_recording_still_lets_the_user_delete_the_figure(rig):
    rig.close_recording()
    assert rig.owner.delete(rig.key) is True, rig.failures
    assert rig.owner.get(rig.key) is None
    assert rig.owner.current_id() != rig.key


def test_the_figure_keeps_its_saved_spectrum_when_its_recording_closes(rig, qtbot):
    rig.close_recording()
    page = rig.page
    np.testing.assert_array_equal(rig.saved_spectrum(), rig.spectrum)
    assert page.has_data() and page.curve_item() is not None
    # What the project holds is that spectrum: saved, put away and brought back from the record.
    assert rig.owner.checkpoint(rig.key)
    assert rig.owner.suspend(rig.key) and rig.owner.page(rig.key) is None
    assert rig.owner.activate(rig.key) is True, rig.failures
    np.testing.assert_array_equal(rig.saved_spectrum(rig.owner.page(rig.key)), rig.spectrum)
    # ...and so is a project written in that state and opened again (neither recording is in it).
    rig.window._close_document(rig.second)
    path = rig.tmp_path / "closed.nanoscope"
    assert rig.window._write_project(str(path)) is True
    second = MainWindow(recovery_enabled=False)
    qtbot.addWidget(second)
    with qtbot.waitSignal(second.projectRestored, timeout=15000):
        second._open_project(str(path))
    restored = second._figure_workspace
    assert restored.activate(rig.key) is True
    np.testing.assert_array_equal(restored.page(rig.key).psd_result()[1], rig.spectrum)
    second._dirty = False


def test_the_figure_says_its_recording_is_not_open(rig):
    rig.close_recording()
    pending = rig.page.inspector_adapter().pending()
    assert pending.count == 1 and pending.reasons == ("selected recording unavailable",)
    assert rig.window._analysis_explorer_status(rig.key, rig.page)[0] == "STALE"
    H.press_update(rig.explorer)
    assert "is not open" in rig.window.statusBar().currentMessage()
    assert rig.failures and "is not open" in rig.failures[-1]
    np.testing.assert_array_equal(rig.saved_spectrum(), rig.spectrum)
    assert rig.owner.get(rig.key).spec.result.status == "stale"


def test_reopening_the_recording_lets_update_recompute_the_figure(rig):
    rig.close_recording()
    reopened = _install(rig.window, _doc(rig.tmp_path, "A.dat", units="pA"))
    assert reopened != rig.first
    rig.window._on_selection(rig.second, 0)
    rig.owner.activate(rig.key)
    H.press_update(rig.explorer)
    assert rig.page.curves()[0].doc_id == reopened
    assert rig.owner.get(rig.key).spec.result.status == "current"
    assert rig.page.inspector_adapter().pending().count == 0


# -- N2 at the page: the same state without the application around it -------------------------------

def _managed_page(qtbot, session, doc_id):
    page = PSDWindow(session, defer_initial_refresh=True)
    qtbot.addWidget(page)
    page._nanoscope_figure_managed = True
    page.set_target(doc_id, 0)
    return page


@pytest.fixture
def two(qapp, tmp_path):
    session = Session()
    first = session.add_document(_doc(tmp_path, "A.dat", units="pA"))
    second = session.add_document(_doc(tmp_path, "B.dat", units="pA", seed=2))
    return session, first, second


def test_a_page_whose_recording_closed_still_captures_what_it_shows(qtbot, two):
    session, first, _second = two
    page = _managed_page(qtbot, session, first)
    key, before = page._pending_target_key(), page.psd_result()[1].copy()
    assert key.endswith("A.dat#0")
    session.close_document(first)

    artifact = capture_psd(page)

    assert artifact.payload["config"]["target_key"] == key
    assert artifact.payload["pending_analysis"] is None, "closing the recording is no draft"
    assert [curve["key"] for curve in artifact.payload["curves"]] == [key]
    np.testing.assert_array_equal(page.psd_result()[1], before)
    assert page._pending_target_key() == key


def test_a_draft_edit_and_a_closed_recording_capture_the_applied_result_and_keep_the_draft(
        qtbot, two):
    session, first, _second = two
    page = _managed_page(qtbot, session, first)
    key, before = page._pending_target_key(), page.psd_result()[1].copy()
    page._nperseg_combo.setCurrentText("1024")          # a draft: nothing is recomputed
    session.close_document(first)

    artifact = capture_psd(page)

    assert artifact.payload["config"]["target_key"] == key
    assert artifact.payload["config"]["nperseg"] == "Auto", "the applied value is what is saved"
    draft = artifact.payload["pending_analysis"]["inputs"]
    assert draft["nperseg"] == "1024" and draft["target_key"] == key
    np.testing.assert_array_equal(page.psd_result()[1], before)
    assert page._nperseg_combo.currentText() == "1024", "the draft is still the user's"
    assert page._pending_target_key() == key


def test_a_saved_figure_restores_with_its_recording_closed(qtbot, tmp_path, two):
    session, first, _second = two
    page = _managed_page(qtbot, session, first)
    before = page.psd_result()[1].copy()
    session.close_document(first)
    artifact = capture_psd(page)

    reopened = _managed_page(qtbot, session, None)
    report = restore_psd(reopened, artifact)

    assert not report.skipped
    np.testing.assert_array_equal(reopened.psd_result()[1], before)
    assert reopened._pending_target_key() == page._pending_target_key()
    assert "selected recording unavailable" in reopened.inspector_adapter().pending().reasons
    assert reopened.inspector_adapter().update() is False


def test_closing_a_compared_recording_still_captures(qtbot, two):
    session, first, second = two
    page = _managed_page(qtbot, session, first)
    page._compare_list.item(0).setCheckState(QtCore.Qt.Checked)
    page.inspector_adapter().update()
    assert [curve.doc_id for curve in page.curves()] == [first, second]
    before = [curve.psd.copy() for curve in page.curves()]
    page._nperseg_combo.setCurrentText("1024")
    session.close_document(second)

    artifact = capture_psd(page)

    assert len(artifact.payload["curves"]) == 2
    for curve, expected in zip(page.curves(), before):
        np.testing.assert_array_equal(curve.psd, expected)


def test_an_open_recording_is_unchanged_by_all_of_this(qtbot, two):
    session, first, second = two
    page = _managed_page(qtbot, session, first)
    key = page._pending_target_key()
    assert page._current_target() == (first, 0) and key.endswith("A.dat#0")
    page._select_combo_target((second, 0))
    assert page._pending_target_key().endswith("B.dat#0")
    page._pending_target(key)
    assert page._current_target() == (first, 0)
    page._pending_target("")
    assert page._pending_target_key() == ""
    page._pending_target("/no/such/recording.dat#0")
    assert page._pending_target_key() == "/no/such/recording.dat#0"


# -- N2, after the review: a recording that comes back, and the closed recording's Source row -----

_SEGMENT_AND_BAND = (0.5, 1.5, 100.0, 1000.0)       # Start s, End s, RMS band low Hz, high Hz


def _four(page):
    """The page's Start, End and RMS band, as the user reads them."""
    return (page._start_spin.value(), page._end_spin.value(),
            page._rms_lo_spin.value(), page._rms_hi_spin.value())


def _recording(tmp_path, name, *, n=100_000, fs=50_000.0):
    """The suite's white-noise current recording, ``n`` samples at ``fs`` (2 s at 50 kHz)."""
    doc = _doc(tmp_path, name, units="pA")
    return replace(doc, meta=replace(doc.meta, n_samples=n, sample_rate_hz=fs),
                   channels=(np.resize(doc.channels[0], n),))


def _source_rows(page):
    """``(text, enabled, (doc_id, channel) or None)`` for every row of the page's Source box."""
    combo = page._rec_combo
    return [(combo.itemText(i), combo.model().item(i).isEnabled(), combo.itemData(i))
            for i in range(combo.count())]


def _row_of(page, doc_id):
    combo = page._rec_combo
    return next(i for i in range(combo.count()) if combo.itemData(i) == (doc_id, 0))


def _reopen_and_update(rig, recording):
    """Open ``recording`` again (a new load), come back to the figure and press Update plot."""
    reopened = _install(rig.window, recording)
    assert reopened != rig.first
    rig.window._on_selection(rig.second, 0)
    rig.owner.activate(rig.key)
    H.press_update(rig.explorer)
    return reopened


def test_a_reopened_recording_keeps_the_figures_segment_and_band_through_update(rig):
    rig.apply(*_SEGMENT_AND_BAND)
    applied = rig.page.curves()[0]
    assert (applied.i0, applied.n_samples) == (25_000, 50_000)
    rig.close_recording()

    reopened = _reopen_and_update(rig, _recording(rig.tmp_path, "A.dat"))

    assert _four(rig.page) == _SEGMENT_AND_BAND, "Update replaced the figure's values"
    curve = rig.page.curves()[0]
    assert (curve.doc_id, curve.i0, curve.n_samples) == (reopened, 25_000, 50_000)
    assert rig.owner.get(rig.key).spec.result.status == "current"
    assert rig.page.inspector_adapter().pending().count == 0
    config = capture_psd(rig.page).payload["config"]
    assert (config["start_s"], config["end_s"],
            config["rms_lo_hz"], config["rms_hi_hz"]) == _SEGMENT_AND_BAND


def test_a_checkpoint_while_the_recording_is_closed_does_not_cost_the_figure_its_values(rig):
    """A draft makes every checkpoint write the applied values back through the page's
    'unavailable recording' route, which used to forget what the four values belonged to."""
    rig.apply(*_SEGMENT_AND_BAND)
    rig.close_recording()
    rig.page._nperseg_combo.setCurrentText("1024")
    assert rig.owner.activate(rig.other) is True and rig.owner.activate(rig.key) is True
    assert rig.page._nperseg_combo.currentText() == "1024"

    _reopen_and_update(rig, _recording(rig.tmp_path, "A.dat"))

    assert _four(rig.page) == _SEGMENT_AND_BAND
    assert rig.page.curves()[0].nperseg == 1024, "the draft was applied by the same Update"


@pytest.mark.parametrize("n, fs", [(150_000, 50_000.0), (50_000, 25_000.0)],
                         ids=["another length", "another rate"])
def test_a_recording_of_another_length_or_rate_under_the_same_key_gets_the_defaults(rig, n, fs):
    rig.apply(*_SEGMENT_AND_BAND)
    rig.close_recording()

    reopened = _reopen_and_update(rig, _recording(rig.tmp_path, "A.dat", n=n, fs=fs))

    assert rig.page.curves()[0].doc_id == reopened
    assert _four(rig.page) == (0.0, n / fs, 0.0, fs / 2.0)


def _reopen_after_another_close(rig, recording):
    """The recording comes back (a new load) and ANOTHER recording closes before Update: the page
    lists what is open again, and the new load is in that list."""
    reopened = _install(rig.window, recording)
    assert reopened != rig.first
    rig.window._on_selection(rig.second, 0)
    rig.window._close_document(rig.second)
    rig.owner.activate(rig.key)
    return reopened


def test_closing_another_recording_after_the_reopen_keeps_the_values_of_the_same_file(rig):
    rig.apply(*_SEGMENT_AND_BAND)
    rig.close_recording()

    reopened = _reopen_after_another_close(rig, _recording(rig.tmp_path, "A.dat"))
    H.press_update(rig.explorer)

    assert rig.page.curves()[0].doc_id == reopened
    assert _four(rig.page) == _SEGMENT_AND_BAND
    assert rig.page.inspector_adapter().pending().count == 0


@pytest.mark.parametrize("n, fs", [(150_000, 50_000.0), (50_000, 25_000.0)],
                         ids=["another length", "another rate"])
def test_closing_another_recording_after_the_reopen_still_gives_a_different_file_the_defaults(
        rig, n, fs):
    """Closing a recording lists what is open again, so a same-named file of another length or
    rate is in the list before Update. Listing it is not taking it for the recording the four
    values were set for: it still gets the defaults."""
    rig.apply(*_SEGMENT_AND_BAND)
    rig.close_recording()

    reopened = _reopen_after_another_close(rig, _recording(rig.tmp_path, "A.dat", n=n, fs=fs))
    H.press_update(rig.explorer)

    assert rig.page.curves()[0].doc_id == reopened
    assert _four(rig.page) == (0.0, n / fs, 0.0, fs / 2.0)


@pytest.mark.parametrize("n, fs", [(150_000, 50_000.0), (50_000, 25_000.0)],
                         ids=["another length", "another rate"])
def test_a_checkpoint_with_a_draft_after_that_close_still_gives_a_different_file_the_defaults(
        rig, n, fs):
    """The same, through the other route that writes the controls back through the Source box:
    a draft makes every checkpoint do it, and the page by then lists the new load."""
    rig.apply(*_SEGMENT_AND_BAND)
    rig.close_recording()
    reopened = _reopen_after_another_close(rig, _recording(rig.tmp_path, "A.dat", n=n, fs=fs))
    rig.page._nperseg_combo.setCurrentText("1024")
    assert rig.owner.activate(rig.other) is True and rig.owner.activate(rig.key) is True
    assert rig.page._nperseg_combo.currentText() == "1024"

    H.press_update(rig.explorer)

    assert rig.page.curves()[0].doc_id == reopened
    assert _four(rig.page) == (0.0, n / fs, 0.0, fs / 2.0)
    assert rig.page.curves()[0].nperseg == 1024, "the draft was applied by the same Update"


@pytest.mark.parametrize("n, fs", [(100_000, 50_000.0), (150_000, 50_000.0), (50_000, 25_000.0)],
                         ids=["the same file", "another length", "another rate"])
def test_listing_a_load_of_the_figures_recording_leaves_it_for_update_to_judge(
        qtbot, tmp_path, two, n, fs):
    """At the page, with nothing to say what a page was told: the page lists a new load of the
    recording its values were set for (here by a close, the page never told of the open); only
    the same file keeps the values through the next Update."""
    session, first, second = two
    page = _managed_page(qtbot, session, first)
    for spin, value in zip((page._start_spin, page._end_spin, page._rms_lo_spin,
                            page._rms_hi_spin), _SEGMENT_AND_BAND):
        spin.setValue(value)
    page.inspector_adapter().update()
    assert _four(page) == _SEGMENT_AND_BAND
    session.close_document(first)
    page.recording_closed(first)
    reopened = session.add_document(_recording(tmp_path, "A.dat", n=n, fs=fs))
    session.close_document(second)

    page.recording_closed(second)               # lists what is open: the new load is in the list

    assert page._current_target() == (reopened, 0), "the page lists the new load as the figure's own"
    page.inspector_adapter().update()
    expected = _SEGMENT_AND_BAND if (n, fs) == (100_000, 50_000.0) else (0.0, n / fs, 0.0, fs / 2.0)
    assert _four(page) == expected


def test_another_recording_chosen_after_the_close_gets_its_own_defaults(rig):
    rig.apply(*_SEGMENT_AND_BAND)
    rig.close_recording()
    rig.page._rec_combo.setCurrentIndex(_row_of(rig.page, rig.second))
    assert _four(rig.page) == (0.0, 2.0, 0.0, 25_000.0)


@pytest.mark.parametrize("n, fs, expected", [
    (100_000, 50_000.0, _SEGMENT_AND_BAND),
    (75_000, 50_000.0, _SEGMENT_AND_BAND),             # ends exactly where the saved segment does
    (74_000, 50_000.0, (0.0, 1.48, 0.0, 25_000.0)),    # ends before it
    (50_000, 50_000.0, (0.0, 1.0, 0.0, 25_000.0)),
    (100_000, 25_000.0, (0.0, 4.0, 0.0, 12_500.0)),    # long enough: only its rate tells it apart
    (50_000, 25_000.0, (0.0, 2.0, 0.0, 12_500.0)),
], ids=["the same recording", "as long as the saved segment", "just shorter than it",
        "half as long", "another rate", "another rate and shorter"])
def test_a_project_reopened_without_the_recording_keeps_the_values_when_it_is_opened(
        rig, qtbot, n, fs, expected):
    """The saved figure knows its recording's key, the rate its spectrum was computed at and how
    far into the recording that spectrum reached (a project stores no length): the same
    recording keeps the values; another rate, or a recording that ends before the stretch the
    spectrum used, gets the defaults. (A longer file cannot be told apart from the saved figure.)"""
    rig.apply(*_SEGMENT_AND_BAND)
    rig.close_recording()
    rig.window._close_document(rig.second)
    path = rig.tmp_path / "without.nanoscope"
    assert rig.window._write_project(str(path)) is True
    second = MainWindow(recovery_enabled=False)
    qtbot.addWidget(second)
    with qtbot.waitSignal(second.projectRestored, timeout=15000):
        second._open_project(str(path))
    owner = second._figure_workspace
    assert owner.activate(rig.key) is True
    page = owner.page(rig.key)
    assert _four(page) == _SEGMENT_AND_BAND and "selected recording unavailable" in (
        page.inspector_adapter().pending().reasons)

    reopened = _install(second, _recording(rig.tmp_path, "A.dat", n=n, fs=fs))
    assert [(text, enabled) for text, enabled, _data in _source_rows(page)] == [("A.dat", True)], (
        "the box still reads 'Unavailable recording' for a recording that is open")
    second._analysis_explorer.update_current()

    assert _four(page) == expected
    curve = page.curves()[0]
    assert curve.doc_id == reopened and owner.get(rig.key).spec.result.status == "current"
    if expected == _SEGMENT_AND_BAND:
        assert (curve.i0, curve.n_samples) == (25_000, 50_000)
    second._dirty = False


def test_the_source_box_shows_the_closed_recording_as_the_one_unavailable_row(rig):
    rig.close_recording()
    rows = _source_rows(rig.page)
    unavailable = [row for row in rows if row[0].startswith("Unavailable recording")]
    assert len(unavailable) == 1 and not unavailable[0][1], rows
    assert rig.page._rec_combo.currentText() == unavailable[0][0], "the figure's own recording"
    assert [text for text, enabled, _data in rows if enabled] == ["B.dat"], rows
    assert all(data != (rig.first, 0) for _text, _enabled, data in rows), rows


def test_a_recording_that_opens_again_is_offered_by_the_source_box_at_once(rig):
    """The closed recording's row reads 'Unavailable recording'; once the file is open again the
    box must not go on saying so until Update: it offers the new load, and nothing else moves."""
    rig.apply(*_SEGMENT_AND_BAND)
    rig.close_recording()
    assert any(text.startswith("Unavailable recording") for text, _e, _d in _source_rows(rig.page))
    token = rig.owner.revision_token(rig.key)

    reopened = _install(rig.window, _recording(rig.tmp_path, "A.dat"))

    rows = _source_rows(rig.page)
    assert [(text, enabled) for text, enabled, _data in rows] == [
        ("B.dat", True), ("A.dat", True)], rows
    assert rig.page._rec_combo.currentData() == (reopened, 0), "the figure's own recording"
    assert _four(rig.page) == _SEGMENT_AND_BAND
    assert rig.owner.revision_token(rig.key) == token, "opening a recording edited the figure"
    assert rig.page.inspector_adapter().pending().count == 0

    H.press_update(rig.explorer)
    assert [(text, enabled) for text, enabled, _data in _source_rows(rig.page)] == [
        ("B.dat", True), ("A.dat", True)]
    assert _four(rig.page) == _SEGMENT_AND_BAND
    assert rig.page.curves()[0].doc_id == reopened


def test_a_comparison_recording_that_opens_again_is_offered_and_stays_ticked(rig):
    page = rig.page
    page._compare_list.item(0).setCheckState(QtCore.Qt.Checked)
    H.press_update(rig.explorer)
    (key,) = page.compare_keys()
    rig.window._close_document(rig.second)
    assert page._compare_list.item(0).text().startswith("Unavailable recording")

    again = _install(rig.window, _doc(rig.tmp_path, "B.dat", units="pA", seed=5))

    rows = [(page._compare_list.item(i).text(), page._compare_list.item(i).checkState())
            for i in range(page._compare_list.count())]
    assert rows == [("B.dat", QtCore.Qt.Checked)], rows
    assert page.compare_keys() == [key] and page._compare_list.item(0).data(QtCore.Qt.UserRole) == key
    assert [data for _text, _enabled, data in _source_rows(page)] == [(rig.first, 0), (again, 0)]
    assert len(page.curves()) == 2


def test_a_recording_the_page_never_named_still_waits_for_update(rig):
    """Only a recording the page names as unavailable is listed on an open: any other one opened
    since is picked up by Update, as it always was."""
    rig.close_recording()
    before = _source_rows(rig.page)

    _install(rig.window, _doc(rig.tmp_path, "C.dat", units="pA", seed=9))

    assert _source_rows(rig.page) == before
    assert [row[0] for row in before if row[1]] == ["B.dat"]


def test_closing_the_recording_is_no_edit_of_the_figure(rig):
    """Re-listing the recordings on a close moves no value and sends no signal the figure
    workspace would take for an edit (it would mark the project changed)."""
    rig.apply(*_SEGMENT_AND_BAND)
    rig.window._dirty = False
    token = rig.owner.revision_token(rig.key)
    rig.close_recording()
    assert rig.window._dirty is False, "closing a recording marked the project changed"
    assert rig.owner.revision_token(rig.key) == token
    assert _four(rig.page) == _SEGMENT_AND_BAND
    pending = rig.page.inspector_adapter().pending()
    assert (pending.count, pending.reasons) == (1, ("selected recording unavailable",))


def test_a_draft_typed_before_the_close_is_still_the_users_after_it(rig):
    rig.apply(*_SEGMENT_AND_BAND)
    rig.page._start_spin.setValue(0.7)
    rig.page._nperseg_combo.setCurrentText("1024")
    assert rig.page.inspector_adapter().pending().reasons == ("start s", "nperseg")

    rig.close_recording()

    assert _four(rig.page) == (0.7, 1.5, 100.0, 1000.0)
    assert rig.page._nperseg_combo.currentText() == "1024"
    assert rig.page.inspector_adapter().pending().reasons == (
        "start s", "nperseg", "selected recording unavailable")


@pytest.mark.parametrize("ticked", [True, False], ids=["ticked", "not ticked"])
def test_a_closed_comparison_recording_is_not_offered_as_a_live_row(rig, ticked):
    page = rig.page
    if ticked:
        page._compare_list.item(0).setCheckState(QtCore.Qt.Checked)
        H.press_update(rig.explorer)
        assert [curve.doc_id for curve in page.curves()] == [rig.first, rig.second]
    key = page._compare_list.item(0).data(QtCore.Qt.UserRole)
    before = _four(page)

    rig.window._close_document(rig.second)

    rows = [(page._compare_list.item(i).text(), page._compare_list.item(i).checkState())
            for i in range(page._compare_list.count())]
    if ticked:
        assert page.compare_keys() == [key]
        assert [(text.startswith("Unavailable recording"), state) for text, state in rows] == [
            (True, QtCore.Qt.Checked)], rows
    else:
        assert rows == [] and page.compare_keys() == []
    assert [text for text, _enabled, _data in _source_rows(page)] == ["A.dat"]
    assert _four(page) == before and page.current_view() == VIEW_PSD


def test_a_checkpoint_with_a_draft_leaves_one_row_for_the_closed_recording(rig):
    rig.close_recording()
    rig.page._rec_combo.setCurrentIndex(_row_of(rig.page, rig.second))     # a draft
    assert rig.owner.activate(rig.other) is True and rig.owner.activate(rig.key) is True
    rows = _source_rows(rig.page)
    assert sum("A.dat" in text for text, _enabled, _data in rows) == 1, rows


def test_no_row_the_user_can_still_pick_zeroes_the_values_after_the_close(rig):
    rig.apply(*_SEGMENT_AND_BAND)
    rig.close_recording()
    combo = rig.page._rec_combo
    combo.setCurrentIndex(_row_of(rig.page, rig.second))
    chosen = _four(rig.page)
    assert chosen == (0.0, 2.0, 0.0, 25_000.0)
    for index, (_text, enabled, _data) in enumerate(_source_rows(rig.page)):
        if enabled:
            combo.setCurrentIndex(index)
            assert _four(rig.page) == chosen, combo.itemText(index)


def test_choosing_the_unavailable_row_leaves_the_values_alone(rig):
    rig.apply(*_SEGMENT_AND_BAND)
    rig.close_recording()
    combo = rig.page._rec_combo
    combo.setCurrentIndex(_row_of(rig.page, rig.second))
    chosen = _four(rig.page)
    unavailable = next(i for i, (text, _enabled, _data) in enumerate(_source_rows(rig.page))
                       if text.startswith("Unavailable recording"))
    combo.setCurrentIndex(unavailable)
    assert _four(rig.page) == chosen, "the unavailable row zeroed the segment and band"


def test_a_closed_recordings_row_never_zeroes_the_segment_and_band(qtbot, two):
    """A page nobody told about the close still lists the recording: its row must not zero."""
    session, first, second = two
    page = _managed_page(qtbot, session, first)
    stale = _row_of(page, first)
    session.close_document(first)
    page._rec_combo.setCurrentIndex(_row_of(page, second))
    chosen = _four(page)
    assert chosen == (0.0, 2.0, 0.0, 25_000.0)

    page._rec_combo.setCurrentIndex(stale)

    assert _four(page) == chosen, "the closed recording's row zeroed the segment and band"
    assert page._start_spin.maximum() == 2.0 and page._rms_hi_spin.maximum() == 25_000.0


def test_with_no_recording_at_all_the_segment_and_band_still_read_zero(qtbot, tmp_path):
    session = Session()
    only = session.add_document(_doc(tmp_path, "A.dat", units="pA"))
    page = _managed_page(qtbot, session, only)
    session.close_document(only)
    page.refresh()
    assert page._rec_combo.count() == 0 and _four(page) == (0.0, 0.0, 0.0, 0.0)


# -- N2, after the review: a comparison closed before Save --------------------------------------

def _through_a_project(tmp_path, artifact, name):
    """What Save Project then Open Project does to one Noise artifact."""
    path = tmp_path / f"{name}.nanoscope"
    save_project(path, ProjectState(workspace=AnalyticalWorkspace(artifacts=[artifact])))
    return load_project(path).workspace.artifacts[0]


def _assert_comparison_kept(page, key):
    assert page.compare_keys() == [key]
    (row,) = [page._compare_list.item(i) for i in range(page._compare_list.count())
              if page._compare_list.item(i).data(QtCore.Qt.UserRole) == key]
    assert row.text().startswith("Unavailable recording") and row.checkState() == QtCore.Qt.Checked
    assert len(page.curves()) == 2, "the saved figure lost a curve"


def test_a_comparison_closed_before_save_is_still_a_comparison_after_two_saves_and_reopens(
        qtbot, tmp_path, two):
    session, first, second = two
    page = _managed_page(qtbot, session, first)
    page._compare_list.item(0).setCheckState(QtCore.Qt.Checked)
    page.inspector_adapter().update()
    (key,) = page.compare_keys()
    assert key.endswith("B.dat#0") and len(page.curves()) == 2
    session.close_document(second)

    artifact = _through_a_project(tmp_path, capture_psd(page), "first")            # Save, ...
    reopened = _managed_page(qtbot, session, None)
    assert not restore_psd(reopened, artifact).skipped                            # ...Open
    _assert_comparison_kept(reopened, key)

    artifact = _through_a_project(tmp_path, capture_psd(reopened), "second")       # Save, ...
    again = _managed_page(qtbot, session, None)
    assert not restore_psd(again, artifact).skipped                               # ...Open
    _assert_comparison_kept(again, key)
    for sources in (capture_psd(page).payload["config"]["compare_sources"],
                    capture_psd(again).payload["config"]["compare_sources"]):
        assert [source["source_key"] for source in sources] == [key.rsplit("#", 1)[0]], sources


def test_a_project_saved_before_the_repair_still_opens_and_heals_on_its_next_save(
        qtbot, tmp_path, two):
    """The old build saved a closed comparison with an EMPTY identity (and its tick in
    ``compare_keys``). Such a project opens as it did, and the next save names the recording."""
    session, first, second = two
    page = _managed_page(qtbot, session, first)
    page._compare_list.item(0).setCheckState(QtCore.Qt.Checked)
    page.inspector_adapter().update()
    (key,) = page.compare_keys()
    session.close_document(second)
    artifact = capture_psd(page)
    empty = {"source_key": "", "channel": 0, "channel_name": None, "channel_kind": None}
    old = replace(artifact, payload={**artifact.payload, "config": {
        **artifact.payload["config"], "compare_sources": [empty]}})

    reopened = _managed_page(qtbot, session, None)
    assert not restore_psd(reopened, _through_a_project(tmp_path, old, "old")).skipped
    _assert_comparison_kept(reopened, key)
    healed = capture_psd(reopened)
    assert [source["source_key"] for source in healed.payload["config"]["compare_sources"]] == [
        key.rsplit("#", 1)[0]]

    again = _managed_page(qtbot, session, None)
    assert not restore_psd(again, _through_a_project(tmp_path, healed, "healed")).skipped
    _assert_comparison_kept(again, key)


def test_a_comparison_closed_in_the_app_survives_two_saves_and_reopens(rig, qtbot):
    """The same through File ▸ Save Project and Open Project, on the real windows: here the
    page has been told about the close, so it no longer lists the comparison at all."""
    page = rig.page
    page._compare_list.item(0).setCheckState(QtCore.Qt.Checked)
    H.press_update(rig.explorer)
    (key,) = page.compare_keys()
    assert len(page.curves()) == 2
    rig.window._close_document(rig.second)
    _assert_comparison_kept(page, key)
    rig.close_recording()                   # (no file stands behind these synthetic recordings)

    window = rig.window
    for cycle in (1, 2):
        path = rig.tmp_path / f"cycle{cycle}.nanoscope"
        assert window._write_project(str(path)) is True
        window = MainWindow(recovery_enabled=False)
        qtbot.addWidget(window)
        with qtbot.waitSignal(window.projectRestored, timeout=15000):
            window._open_project(str(path))
        assert window._figure_workspace.activate(rig.key) is True
        _assert_comparison_kept(window._figure_workspace.page(rig.key), key)
        window._dirty = False
