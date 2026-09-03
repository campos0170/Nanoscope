"""The three views added to the power-spectrum window, driven the way the owner drives them.

Not "the widget exists". Every assertion below runs against a REAL
:class:`~nanoscope.gui.main_window.MainWindow` built through
:func:`nanoscope.gui.app.create_app`, sized to the owner's 1512x900 display, with the PSD
window opened from the Analysis menu action and the view changed by KEYBOARD on the combo
— because a control whose row runs past ~1100 px on that screen is not reachable and is
therefore not done, and because a view that only ``set_view()`` can reach is not a view.

The three:

* **FFT** — a single-shot magnitude spectrum of the selected span, which must SAY on screen
  that it is unaveraged and that the Welch PSD beside it is the better noise estimate. The
  spikier curve looks like the more informative one; the caption is what stops that.
* **Spectrogram** — the bigger win for a nanopore user: a burst that starts partway through
  is a lift of a factor of two in the whole-trace Welch average and a bright stripe here,
  at the time it happened. That contrast is measured below on both surfaces.
* **All-points histogram** — the levels, read off with NO detection run: the test asserts
  the Analysis-menu actions that need a detection are still dead while both levels come back.

And two invariants the campaign is under standing orders about: nothing moves under the
pointer when the view changes, and no measured number moves — the Welch PSD's own curve and
summary are hashed before and after a tour of all four views.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import hashlib
import warnings

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtTest, QtWidgets
from PySide6.QtCore import Qt

from nanoscope.core.document import TraceDocument
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.app import create_app
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.psd_panel import (
    VIEW_ALLPOINTS,
    VIEW_FFT,
    VIEW_PSD,
    VIEW_SPECTROGRAM,
    VIEWS,
)

FS = 50_000.0
SCREEN = (1512, 900)
#: A control whose row needs more than this is off the owner's screen. See the campaign note.
REACHABLE_PX = 1100

OPEN_PA = 100.0
BLOCKED_PA = 60.0
BURST_FROM, BURST_TO = 120_000, 150_000        # 2.4-3.0 s of a 5 s record


# ================================================================================================
# builders
# ================================================================================================

def _pore_trace(n: int = 250_000, seed: int = 0) -> np.ndarray:
    """5 s of a pore: open at 100 pA, a 2 000-sample blockade, 60 Hz mains, and a burst.

    Synthetic because every number this file asserts was CHOSEN: the two levels are 100 and
    60 pA, the mains line is 2 pA peak at 60 Hz, and the noise burst covers exactly 2.4-3.0 s.
    """
    rng = np.random.default_rng(seed)
    sig = (rng.normal(0.0, 3.0, n) + OPEN_PA).astype(np.float32)
    sig += (2.0 * np.sin(2 * np.pi * 60.0 * np.arange(n) / FS)).astype(np.float32)
    sig[BURST_FROM:BURST_TO] += rng.normal(0.0, 12.0, BURST_TO - BURST_FROM).astype(np.float32)
    sig[60_000:62_000] -= (OPEN_PA - BLOCKED_PA)
    return sig


def _doc(tmp_path, sig: np.ndarray, name: str = "pore.dat") -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=int(sig.size),
        sample_rate_hz=FS, rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


@pytest.fixture
def opened(qapp, qtbot, tmp_path):
    """A real MainWindow at 1512x900 with the PSD window opened from the Analysis action."""
    create_app([])                       # the shipped app's own QApplication setup
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(*SCREEN)
    win.show()
    doc = _doc(tmp_path, _pore_trace())
    doc_id = win._session.add_document(doc)
    win._create_default_layers(doc)
    win._on_selection(doc_id, 0)
    win._sync_psd_action()
    assert win._act_psd.isEnabled(), "the menu action must be live for a current channel"

    win._act_psd.trigger()               # the user's own route in
    psd = win._psd_window
    qtbot.addWidget(psd)
    psd.resize(1080, 720)                # its own default, which fits 1512x900
    psd.show()
    qapp.processEvents()
    return win, psd, doc_id


def _select_view_by_keyboard(qapp, psd, view: str) -> None:
    """Walk the View combo with the Down/Up key until ``view`` is current.

    Keyboard rather than ``set_view`` on purpose: this is the path a user has, and it also
    proves the combo is focusable and enabled rather than merely present.
    """
    want = next(i for i, (key, _) in enumerate(VIEWS) if key == view)
    psd._view_combo.setFocus()
    guard = 0
    while psd._view_combo.currentIndex() != want and guard < 10:
        key = Qt.Key_Down if psd._view_combo.currentIndex() < want else Qt.Key_Up
        QtTest.QTest.keyClick(psd._view_combo, key)
        guard += 1
    qapp.processEvents()
    assert psd.current_view() == view


def _row_right_edge(psd, widget: QtWidgets.QWidget) -> int:
    """Where this control's row ENDS in the window, in px from the window's left edge."""
    return widget.mapTo(psd, widget.rect().topRight()).x()


def _control_ys(psd) -> dict:
    return {name: getattr(psd, attr).mapTo(psd, getattr(psd, attr).rect().topLeft()).y()
            for name, attr in (("segment start", "_start_spin"),
                               ("welch nperseg", "_nperseg_combo"),
                               ("log-log", "_loglog_check"),
                               ("rms low edge", "_rms_lo_spin"),
                               ("save button", "_save_btn"))}


# ================================================================================================
# reachability + stability on the owner's screen
# ================================================================================================

def test_every_added_control_is_reachable_on_the_1512x900_display(opened, qapp):
    """Each new control's row ends far inside the 1100 px an unreachable control needs.

    The View selector and all three views' options live in the same 300-430 px control
    column the Segment and Welch groups already use, so none of them can run off the screen
    the way a hardcoded 1600 px window does.
    """
    _win, psd, _doc_id = opened
    controls = [psd._view_combo, psd._spec_nperseg_combo, psd._spec_cmap_combo,
                psd._hist_bins_combo, psd._hist_clip_check, psd._hist_levels_check]
    for widget in controls:
        edge = _row_right_edge(psd, widget)
        assert 0 < edge <= REACHABLE_PX, f"{widget} row ends at {edge} px"
    assert psd.width() <= SCREEN[0] and psd.height() <= SCREEN[1]


def test_changing_the_view_moves_no_control_under_the_pointer(opened, qapp):
    """The five control rows below the View group sit at the SAME y in all four views.

    Both stacks (the plot area and the per-view options) reserve their tallest page, and the
    options stack's height is pinned to a measured maximum — a word-wrapped note whose
    one-line size hint became three laid-out lines had already pushed these rows 20 px down
    in the FFT view before the height was fixed.
    """
    _win, psd, _doc_id = opened
    baseline = None
    for key, _label in VIEWS:
        _select_view_by_keyboard(qapp, psd, key)
        ys = _control_ys(psd)
        if baseline is None:
            baseline = ys
        else:
            assert ys == baseline, f"controls moved in the {key} view: {ys} != {baseline}"


def test_no_measured_number_moves_when_the_added_views_are_visited(opened, qapp):
    """A tour of all four views leaves the Welch PSD's own numbers bit-identical.

    These views ADD readings; they do not change a measurement. The primary curve's
    frequencies and PSD are hashed, and the noise summary compared, before and after.
    """
    _win, psd, _doc_id = opened
    before_hash = hashlib.sha256(
        np.ascontiguousarray(psd.psd_result()[0]).tobytes()
        + np.ascontiguousarray(psd.psd_result()[1]).tobytes()).hexdigest()
    before_rms = psd.summary()["total_rms"]

    for key, _label in VIEWS:
        _select_view_by_keyboard(qapp, psd, key)
    _select_view_by_keyboard(qapp, psd, VIEW_PSD)

    after_hash = hashlib.sha256(
        np.ascontiguousarray(psd.psd_result()[0]).tobytes()
        + np.ascontiguousarray(psd.psd_result()[1]).tobytes()).hexdigest()
    assert after_hash == before_hash
    assert psd.summary()["total_rms"] == before_rms


# ================================================================================================
# (1) FFT
# ================================================================================================

def test_the_fft_view_draws_the_span_and_finds_the_mains_line(opened, qapp):
    """A real curve on a real axis, with the 2 pA 60 Hz line at its own amplitude."""
    _win, psd, _doc_id = opened
    _select_view_by_keyboard(qapp, psd, VIEW_FFT)

    spec = psd.fft_spectrum()
    assert spec is not None and spec.n_samples == 250_000
    assert isinstance(psd._fft_item, pg.PlotDataItem)
    xs, ys = psd._fft_item.getData()
    assert xs.size > 100

    near = np.argmin(np.abs(spec.freqs - 60.0))
    assert spec.freqs[near] == pytest.approx(60.0, abs=0.5)
    assert spec.mag[near] == pytest.approx(2.0, rel=0.1)      # the peak amplitude, in pA
    assert "peak amplitude (pA)" in psd._fft_plot.getAxis("left").labelString()


def test_the_fft_view_says_on_screen_that_it_is_the_noisier_estimate(opened, qapp):
    """The honesty requirement, asserted as text the user actually sees.

    An unaveraged periodogram LOOKS more informative than the smooth Welch curve beside it.
    The caption must name it a periodogram, say the uncertainty does not fall with record
    length, and point at the PSD view for noise characterisation.
    """
    _win, psd, _doc_id = opened
    _select_view_by_keyboard(qapp, psd, VIEW_FFT)

    caption = psd.caption_text().lower()
    assert "periodogram" in caption
    assert "unaveraged" in caption
    assert "welch psd view is the better tool" in caption
    assert "envelope" in caption          # and it says the drawn curve was reduced


# ================================================================================================
# (2) spectrogram
# ================================================================================================

def test_the_spectrogram_shows_the_burst_that_the_welch_average_only_hints_at(opened, qapp):
    """The claim the audit rated the bigger loss, measured on both surfaces at once.

    The same 0.6 s of extra noise: a factor of ~2 lift in the whole-trace Welch PSD with no
    indication of WHEN, against a >5x brighter block of columns here, located at 2.4-3.0 s.
    """
    _win, psd, doc_id = opened
    _select_view_by_keyboard(qapp, psd, VIEW_SPECTROGRAM)
    sg = psd.spectrogram()
    assert sg is not None

    band = slice(sg.freqs.size // 4, None)         # the mid/high decades the burst lives in
    per_column = np.median(sg.psd[band, :], axis=0)
    in_burst = (sg.times >= BURST_FROM / FS) & (sg.times <= BURST_TO / FS)
    quiet = sg.times < (BURST_FROM / FS) * 0.9
    assert in_burst.any() and quiet.any()
    contrast = float(np.median(per_column[in_burst]) / np.median(per_column[quiet]))
    assert contrast > 5.0

    # ...while the whole-trace Welch estimate the PSD view shows is only mildly lifted, and
    # says nothing at all about when.
    from nanoscope.analysis.psd import welch_psd
    data = np.asarray(psd._session.calibrated(doc_id, 0).data)
    whole = welch_psd(data, FS, nperseg=sg.nperseg)[1]
    calm = welch_psd(data[:BURST_FROM], FS, nperseg=sg.nperseg)[1]
    lift = float(np.median(whole[band]) / np.median(calm[band]))
    assert 1.2 < lift < contrast / 2.0

    # and the image is really on the plot, at the recording's own clock
    assert psd._spec_image.image is not None and psd._spec_image.image.size > 1
    rect = psd._spec_image.boundingRect()
    assert rect.width() > 0 and rect.height() > 0


def test_the_spectrogram_caption_states_how_wide_one_column_is(opened, qapp):
    """Time resolution is the number that decides whether a burst is resolvable at all."""
    _win, psd, _doc_id = opened
    _select_view_by_keyboard(qapp, psd, VIEW_SPECTROGRAM)
    caption = psd.caption_text()
    sg = psd.spectrogram()
    assert f"{sg.freqs.size:,} frequency bins" in caption
    assert "time columns" in caption
    assert "ms)" in caption               # one column's width in milliseconds


def test_the_spectrogram_time_resolution_control_changes_the_picture(opened, qapp):
    """The view's own nperseg is a live control, not a decoration."""
    _win, psd, _doc_id = opened
    _select_view_by_keyboard(qapp, psd, VIEW_SPECTROGRAM)
    auto = psd.spectrogram()

    psd._spec_nperseg_combo.setCurrentText("4096")
    qapp.processEvents()
    fine_in_frequency = psd.spectrogram()

    assert fine_in_frequency.nperseg == 4096 != auto.nperseg
    assert fine_in_frequency.freqs.size > auto.freqs.size          # finer in frequency
    assert fine_in_frequency.times.size < auto.times.size          # coarser in time


def test_an_unusable_spectrogram_clears_the_previous_recordings_chrome(opened, qapp, tmp_path):
    """Switching to poisoned samples must not leave the prior title/colour scale behind.

    The caption already named the NaNs, but the early-return path retained both
    ``Spectrogram — pore.dat`` and that recording's live colour bar. Two surfaces then
    identified one empty plot as two different recordings.
    """
    win, psd, good_id = opened
    _select_view_by_keyboard(qapp, psd, VIEW_SPECTROGRAM)
    assert psd._spec_bar is not None and psd._spec_bar.isVisible()

    bad = np.full(20_000, np.nan, dtype=np.float32)
    bad_id = win._session.add_document(_doc(tmp_path, bad, "all-nan.dat"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        psd.set_target(bad_id, 0)
    qapp.processEvents()

    assert "all-nan.dat" in psd._spec_plot.titleLabel.text
    assert "pore.dat" not in psd._spec_plot.titleLabel.text
    assert psd._spec_msg is not None
    assert not psd._spec_image.isVisible()
    assert not psd._spec_bar.isVisible()
    (x0, x1), (y0, y1) = psd._spec_plot.vb.viewRange()
    assert x0 <= 0.5 <= x1 and y0 <= 0.5 <= y1, "the refusal itself is off-screen"

    # The same persistent widgets come back on a valid target; hiding the stale state
    # must not turn into a one-way door.
    psd.set_target(good_id, 0)
    qapp.processEvents()
    assert "pore.dat" in psd._spec_plot.titleLabel.text
    assert psd._spec_msg is None
    assert psd._spec_image.isVisible() and psd._spec_bar.isVisible()


# ================================================================================================
# (3) all-points histogram
# ================================================================================================

def test_the_all_points_view_reads_both_levels_with_no_detection_run(opened, qapp):
    """The point of this view: the open and blocked levels, before any detector exists.

    No detection has run — the Analysis-menu actions that need one are still dead — so
    nothing here can be coming from a detection, a provenance record or the frozen
    20-column schema.
    """
    win, psd, _doc_id = opened
    assert not win._act_plots.isEnabled()            # no detection, and none needed

    _select_view_by_keyboard(qapp, psd, VIEW_ALLPOINTS)
    hist = psd.all_points()
    assert hist is not None and hist.n_binned == 250_000

    levels = psd.hist_levels()
    assert len(levels) == 2
    assert levels[0]["center"] == pytest.approx(OPEN_PA, abs=1.0)
    assert levels[1]["center"] == pytest.approx(BLOCKED_PA, abs=2.0)
    # both marked on the plot, so the picture carries the numbers too
    assert len(psd._hist_marks) == 2
    assert psd._hist_plot.getAxis("left").labelString().lower().count("log") == 1


def test_the_histogram_draws_empty_bins_as_gaps_rather_than_inventing_a_floor(opened, qapp):
    """On a log count axis there is no honest place to put a zero."""
    _win, psd, _doc_id = opened
    _select_view_by_keyboard(qapp, psd, VIEW_ALLPOINTS)
    hist = psd.all_points()
    # getData() returns what the LOG axis draws, so the comparison is against log10(counts)
    _xs, ys = psd._hist_item.getData()
    empty = hist.counts == 0
    assert empty.any(), "this trace has empty bins — otherwise the test proves nothing"
    assert np.isnan(ys[empty]).all()
    np.testing.assert_allclose(ys[~empty], np.log10(hist.counts[~empty].astype(float)))


def test_the_bin_count_control_regrids_the_histogram(opened, qapp):
    _win, psd, _doc_id = opened
    _select_view_by_keyboard(qapp, psd, VIEW_ALLPOINTS)
    assert psd.all_points().counts.size == 512
    psd._hist_bins_combo.setCurrentText("2048")
    qapp.processEvents()
    assert psd.all_points().counts.size == 2048
    assert psd.all_points().n_binned == 250_000      # still every sample


# ================================================================================================
# the shared span, and the export
# ================================================================================================

def test_all_three_views_follow_the_segment_spins(opened, qapp):
    """One span, four views. Narrowing the segment narrows every one of them.

    And the spectrogram's time axis is drawn on the SAME recording clock the Segment spins
    take, so a feature at 3.0 s in the picture is at 3.0 s in the spin boxes.
    """
    _win, psd, _doc_id = opened
    psd.set_segment(2.0, 3.0)                        # 50 000 samples
    qapp.processEvents()

    _select_view_by_keyboard(qapp, psd, VIEW_ALLPOINTS)
    assert psd.all_points().n_total == 50_000

    _select_view_by_keyboard(qapp, psd, VIEW_FFT)
    assert psd.fft_spectrum().n_samples == 50_000

    _select_view_by_keyboard(qapp, psd, VIEW_SPECTROGRAM)
    sg = psd.spectrogram()
    assert sg.n_samples == 50_000
    image = psd._spec_image
    x0 = image.mapRectToView(image.boundingRect()).left()
    assert x0 == pytest.approx(2.0, abs=sg.nperseg / FS)   # recording seconds, not span ones


def test_save_image_writes_the_view_that_is_on_screen(opened, qapp, tmp_path):
    """Pressing Save with a spectrogram showing and receiving a spectrum ends up in a paper."""
    _win, psd, _doc_id = opened
    written = {}
    for view in (VIEW_PSD, VIEW_FFT, VIEW_SPECTROGRAM, VIEW_ALLPOINTS):
        _select_view_by_keyboard(qapp, psd, view)
        out = psd.save_image(str(tmp_path / f"{view}.png"))
        assert out is not None and out.exists() and out.stat().st_size > 1000
        written[view] = out.read_bytes()
    assert len({v[:2048] for v in written.values()}) == 4   # four different pictures
