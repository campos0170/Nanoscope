"""The three views added to the power-spectrum page, driven the way the owner drives them.

Not "the widget exists". Every assertion below runs against a REAL
:class:`~nanoscope.gui.main_window.MainWindow` built through
:func:`nanoscope.gui.app.create_app`, sized to the owner's 1512x900 display, with the Noise
page opened from the Analysis menu action and the view changed by KEYBOARD on the combo
— because a control that runs off that screen is not reachable and is therefore not done,
and because a view that only ``set_view()`` can reach is not a view.

2026-11-11: the Noise page is no longer a free-standing window that computes when it opens.
Since 8661d35 the Analysis menu opens it as a managed figure HOSTED in the Analysis
Explorer: its controls live in the Explorer's inspector, and every analysis control -- the
View selector included -- is a draft until Update plot is pressed. So the fixture presses
Update plot once, a view is chosen by keyboard on the combo AND applied with Update plot,
and geometry is measured in the Explorer, on the controls that are on screen.

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

from . import p1_hosting as H

FS = 50_000.0
SCREEN = (1512, 900)
#: A control whose row needs more than this is off the owner's screen. See the campaign note.
REACHABLE_PX = 1100
#: What the toolbar's Update button reads with nothing pending, and with one change waiting.
UPDATE_IDLE = "Update plot"
UPDATE_ONE_PENDING = "Update plot (1)"

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
    """A real MainWindow at 1512x900 with the Noise page opened from the Analysis action,
    hosted in the Analysis Explorer, and its first spectrum computed with Update plot."""
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
    explorer = win._analysis_explorer
    qapp.processEvents()
    # 2026-11-11 (8661d35): the page is hosted in the Explorer (it used to be its own
    # 1080x720 window, which this fixture resized and showed) and computes nothing until
    # Update plot is pressed -- which is what the user does next.
    assert psd.window() is explorer
    assert not psd.has_data()
    assert explorer.toolbar().update_button.text() == UPDATE_ONE_PENDING
    H.press_update(explorer)
    assert psd.has_data()
    assert explorer.toolbar().update_button.text() == UPDATE_IDLE
    return win, psd, doc_id


def _select_view_by_keyboard(qapp, psd, view: str) -> None:
    """Walk the View combo with the Down/Up key until ``view`` is chosen, then apply it.

    Keyboard rather than ``set_view`` on purpose: this is the path a user has, and it also
    proves the combo is focusable and enabled rather than merely present.

    2026-11-11 (8661d35): on the hosted page the keys only move the combo. The choice is a
    DRAFT -- the page keeps showing the view it had and the toolbar says one change is
    waiting -- until Update plot is pressed, which is the second half of this gesture.
    """
    explorer = psd.window()
    shown = psd.current_view()
    want = next(i for i, (key, _) in enumerate(VIEWS) if key == view)
    psd._view_combo.setFocus()
    guard = 0
    while psd._view_combo.currentIndex() != want and guard < 10:
        key = Qt.Key_Down if psd._view_combo.currentIndex() < want else Qt.Key_Up
        QtTest.QTest.keyClick(psd._view_combo, key)
        guard += 1
    qapp.processEvents()
    assert psd._view_combo.currentIndex() == want        # the keys moved the combo
    if view != shown:
        assert psd.current_view() == shown               # ...and nothing else
        assert explorer.toolbar().update_button.text() == UPDATE_ONE_PENDING
        H.press_update(explorer)
    assert psd.current_view() == view
    assert explorer.toolbar().update_button.text() == UPDATE_IDLE


def _on_screen(psd, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """The Explorer ``widget`` is shown (not hidden) in -- asserted, because only then is a
    position measured in it a measurement. A control that is hidden (another view's option, a
    collapsed group, the Save button a hosted page does not show) keeps whatever geometry
    its layout last gave it, an unlaid 640 px row among them, and says nothing true.

    Shown is all this proves. A shown row can still sit below the bottom, or past the right
    edge, of the scroll area that holds it, where nobody sees it:
    :func:`_visible_right_edge` is what bounds a row on the right."""
    explorer = psd.window()
    assert widget.window() is explorer, f"{widget} is not in the Explorer the page is hosted in"
    assert widget.isVisible(), f"{widget} is not on screen"
    return explorer


def _row_right_edge(psd, widget: QtWidgets.QWidget) -> int:
    """Where this control's row ENDS in the Explorer, in px from the Explorer's left edge
    (2026-11-11: it used to be measured from the page's own window)."""
    explorer = _on_screen(psd, widget)
    return widget.mapTo(explorer, widget.rect().topRight()).x()


def _visible_right_edge(psd, widget: QtWidgets.QWidget) -> int:
    """The last pixel column, in the Explorer's px, at which the inspector shows ``widget``.

    The hosted controls sit in a scroll area (the Data & analysis tab's, or the Format tab's)
    whose horizontal scroll bar is always off: a row laid out past its viewport is clipped
    and cannot be scrolled to, even while it is still inside the Explorer's own window.
    Measured on the same footing as :func:`_row_right_edge` (``topRight``), so a row is whole
    exactly when its edge is no greater than this."""
    explorer = _on_screen(psd, widget)
    area = widget.parentWidget()
    while area is not None and not isinstance(area, QtWidgets.QAbstractScrollArea):
        area = area.parentWidget()
    assert area is not None, f"{widget} is not inside a scroll area"
    viewport = area.viewport()
    return viewport.mapTo(explorer, viewport.rect().topRight()).x()


def _control_ys(psd) -> dict:
    """Each control's top edge in the Explorer, in px from its top.

    2026-11-11: the rows shown in the hosted inspector (the Estimator rows may be below the
    fold; their layout position is still compared). The View selector joins the list, as the
    row the choice is made on; the Save button leaves it, because a hosted page hides it (its
    verbs are the Explorer's Export menu). Of the five, the segment start and the View
    selector sit above the per-view options stack and the other three below it.
    """
    ys = {}
    for name, attr in (("segment start", "_start_spin"),
                       ("view selector", "_view_combo"),
                       ("welch nperseg", "_nperseg_combo"),
                       ("log-log", "_loglog_check"),
                       ("rms low edge", "_rms_lo_spin")):
        widget = getattr(psd, attr)
        explorer = _on_screen(psd, widget)
        ys[name] = widget.mapTo(explorer, widget.rect().topLeft()).y()
    return ys


# ================================================================================================
# reachability + stability on the owner's screen
# ================================================================================================

def test_every_added_control_is_reachable_on_the_1512x900_display(opened, qapp):
    """Each new control's row ends far inside the 1100 px an unreachable control needs.

    The View selector and all three views' options live in the same inspector column the
    Sources and Estimator groups already use, so none of them can run off the screen the
    way a hardcoded 1600 px window does.

    2026-11-11: that column is the Analysis Explorer's, so what is asked is that the
    Explorer needs no more than those 1100 px (its minimum width is 840), fits the display,
    and has every control INSIDE it. The edge is measured in the Explorer, each control
    while the view that shows it is applied -- the options page follows the applied view --
    and the spectrogram's colour-map combo, which moved to the Format tab's Colour map
    panel, in that tab. (The Explorer's own size is not the bound: it is clamped to the
    screen it is on, so it is what the environment makes it.)

    "Inside" is the scroll area's viewport, not only the Explorer's window: the inspector's
    horizontal scroll bar is always off, so a row that ends between the viewport's right
    edge and the window's is clipped, and nobody can scroll to the rest of it.
    """
    _win, psd, _doc_id = opened
    explorer = psd.window()
    assert explorer.minimumWidth() <= REACHABLE_PX
    covered = set()

    def check(name: str, view: str) -> None:
        widget = getattr(psd, name)
        edge = _row_right_edge(psd, widget)
        assert 0 < edge <= explorer.width(), \
            f"{name} row ends at {edge} px, outside the {explorer.width()} px Explorer ({view} view)"
        shown_to = _visible_right_edge(psd, widget)
        assert edge <= shown_to, (
            f"{name} row ends at {edge} px, past the {shown_to} px where the inspector stops "
            f"showing it ({view} view)")
        covered.add(name)

    for view, names in ((VIEW_PSD, ("_view_combo",)),
                        (VIEW_FFT, ("_view_combo",)),
                        (VIEW_SPECTROGRAM, ("_view_combo", "_spec_nperseg_combo")),
                        (VIEW_ALLPOINTS, ("_view_combo", "_hist_bins_combo",
                                          "_hist_clip_check", "_hist_levels_check"))):
        _select_view_by_keyboard(qapp, psd, view)
        for name in names:
            check(name, view)

    _select_view_by_keyboard(qapp, psd, VIEW_SPECTROGRAM)
    H.show_format(explorer, "colour_map")
    assert psd._spec_cmap_combo.isEnabled()              # live in the view it belongs to
    check("_spec_cmap_combo", VIEW_SPECTROGRAM)
    H.show_data(explorer)

    # none of the six the test has always covered was dropped on the way
    assert covered == {"_view_combo", "_spec_nperseg_combo", "_spec_cmap_combo",
                       "_hist_bins_combo", "_hist_clip_check", "_hist_levels_check"}
    assert explorer.width() <= SCREEN[0] and explorer.height() <= SCREEN[1]


def test_changing_the_view_moves_no_control_under_the_pointer(opened, qapp):
    """The three rows below the per-view options stack sit at the SAME y in all four views.

    Both stacks (the plot area and the per-view options) reserve their tallest page, and the
    options stack's height is pinned to a measured maximum — a word-wrapped note whose
    one-line size hint became three laid-out lines had already pushed these rows 20 px down
    in the FFT view before the height was fixed.

    2026-11-11: measured in the Explorer, on the rows shown there (see :func:`_control_ys`).
    Three rows lie below the stack and can be moved by it: the log-log box, the Welch
    segment length and the RMS low edge. (The stand-alone window had five rows below
    the View group.) The segment start and the View selector sit ABOVE the stack, so it
    cannot move them: they are controls, there to show that the measurement itself holds
    still. The Welch and RMS rows sit in the Estimator group, which opens collapsed, so it
    is opened first; they may lie below the fold of the inspector's scroll area, and it is
    their position in the layout that is compared, not whether they are in view. The
    log-log box sits directly under the options stack and is the one that moves if the
    stack stops reserving its tallest page.
    """
    _win, psd, _doc_id = opened
    psd.window().reveal("estimator")
    qapp.processEvents()
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
    # 2026-11-11: choosing it is a draft; the picture on screen is still the old one until
    # Update plot is pressed.
    assert psd.spectrogram().nperseg == auto.nperseg
    assert psd.window().toolbar().update_button.text() == UPDATE_ONE_PENDING
    H.press_update(psd.window())
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
    # 2026-11-11: a draft until Update plot is pressed; the histogram still has its 512 bins.
    assert psd.all_points().counts.size == 512
    assert psd.window().toolbar().update_button.text() == UPDATE_ONE_PENDING
    H.press_update(psd.window())
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
