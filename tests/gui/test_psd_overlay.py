"""Two pores on one noise plot — the comparison the PSD window could not make.

``PSDWindow`` held a SINGLE curve item, so the most common noise-QC question in
the lab ("is this pore quieter than that one?") could only be answered by
flipping between two screenshots. These tests pin the overlay:

* several open recordings draw several coloured curves, named in a legend;
* every curve keeps its OWN noise summary, units and Welch parameters — the
  numbers are reported per curve, never merged into a figure that describes
  neither pore;
* the band-RMS read-out is per curve too, and it separates a quiet pore from a
  noisy one by the right factor;
* a segment is offered like any other recording, and a segment named after its
  parent still gets its own row (the persistent-key identity);
* the legend is readable on the light background as well as the dark one;
* removing the comparison returns the window to exactly its one-curve behaviour.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.psd_panel import PSDWindow

FS = 50_000.0
N = 200_000                # 4 s
PEDESTAL = 100.0
QUIET, NOISY = 2.0, 8.0    # a 4x RMS difference — unmistakable on the plot


# ================================================================================================
# builders
# ================================================================================================

def _doc(tmp_path, name: str, *, sigma: float, fs: float = FS, n: int = N,
         seed: int = 0, units: str = "pA") -> TraceDocument:
    rng = np.random.default_rng(seed)
    sig = (rng.normal(0.0, sigma, n) + PEDESTAL).astype(np.float32)
    info = ChannelInfo(index=0, name="current", raw_units=units, kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n, sample_rate_hz=fs,
        rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


@pytest.fixture
def two_pores(qapp, qtbot, tmp_path):
    """A window pointed at a quiet pore, with a noisy one open beside it."""
    session = Session()
    quiet = session.add_document(_doc(tmp_path, "quiet.dat", sigma=QUIET, seed=1))
    noisy = session.add_document(_doc(tmp_path, "noisy.dat", sigma=NOISY, seed=2))
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(quiet, 0)
    return w, quiet, noisy


# ================================================================================================
# 1. the overlay
# ================================================================================================

def test_a_second_recording_is_offered_for_comparison(two_pores):
    w, _quiet, _noisy = two_pores
    assert w.comparable_labels() == ["noisy.dat"]     # every recording but the primary
    assert w.compare_keys() == []                     # nothing overlaid until asked
    assert len(w.curves()) == 1


def test_ticking_it_draws_a_second_curve_with_a_legend(two_pores):
    w, _quiet, noisy = two_pores
    w.set_compare([noisy])

    curves = w.curves()
    assert len(curves) == 2
    assert w.curve_labels() == ["quiet.dat", "noisy.dat"]
    # Two real curve items, in two distinct colours.
    items = [c.item for c in curves]
    assert all(i is not None for i in items)
    colours = [i.opts["pen"].color().name() for i in items]
    assert len(set(colours)) == 2
    assert all(i in list(w.plot_item().listDataItems()) for i in items)

    labels = w.legend_labels()
    assert any("quiet.dat" in t for t in labels)
    assert any("noisy.dat" in t for t in labels)


def test_the_primary_curve_accessors_still_describe_the_primary(two_pores):
    """No regression for the single-curve API: it mirrors curve 0, always."""
    w, _quiet, noisy = two_pores
    before = w.summary()["total_rms"]
    w.set_compare([noisy])

    assert w.curve_item() is w.curves()[0].item
    assert w.summary()["total_rms"] == pytest.approx(before)
    x, y = w.plotted_data()
    np.testing.assert_allclose(x, w.curves()[0].x)
    np.testing.assert_allclose(y, w.curves()[0].y)


def test_removing_the_comparison_returns_to_one_curve(two_pores):
    w, _quiet, noisy = two_pores
    w.set_compare([noisy])
    assert len(w.curves()) == 2
    w.set_compare([])
    assert len(w.curves()) == 1
    assert w.legend_labels() == [], "stale legend rows must not survive"


# ================================================================================================
# 2. per-curve reporting — the point of the comparison
# ================================================================================================

def test_each_curve_keeps_its_own_noise_summary(two_pores):
    w, _quiet, noisy = two_pores
    w.set_compare([noisy])

    quiet_s = w.curve_summary("quiet.dat")
    noisy_s = w.curve_summary("noisy.dat")
    assert quiet_s is not None and noisy_s is not None
    assert quiet_s["total_rms"] == pytest.approx(QUIET, rel=0.05)
    assert noisy_s["total_rms"] == pytest.approx(NOISY, rel=0.05)
    # The numbers were never merged into one.
    assert noisy_s["total_rms"] / quiet_s["total_rms"] == pytest.approx(NOISY / QUIET, rel=0.1)


def test_the_details_pane_reports_both_curves_under_their_own_headings(two_pores):
    w, _quiet, noisy = two_pores
    w.set_compare([noisy])

    text = w.details_text()
    assert "quiet.dat" in text and "noisy.dat" in text
    assert text.count("Total RMS noise:") == 2, "one full report per curve"
    assert text.count("Mid-log median proxy:") == 2


def test_the_band_rms_readout_is_per_curve(two_pores):
    w, _quiet, noisy = two_pores
    w.set_compare([noisy])
    w.set_rms_band(0.0, FS / 2)

    per = w.band_rms_per_curve()
    assert set(per) == {"quiet.dat", "noisy.dat"}
    assert per["noisy.dat"] / per["quiet.dat"] == pytest.approx(NOISY / QUIET, rel=0.1)

    text = w.rms_readout_text()
    assert "quiet.dat" in text and "noisy.dat" in text
    assert "pA" in text


def test_a_single_curve_readout_is_unchanged(two_pores):
    """The one-pore read-out keeps its original compact form."""
    w, _quiet, _noisy = two_pores
    w.set_rms_band(0.0, 1000.0)
    text = w.rms_readout_text()
    assert "pA" in text and "1000" in text
    assert "quiet.dat" not in text


def test_a_curve_with_a_different_sample_rate_keeps_its_own(qapp, qtbot, tmp_path):
    """Two pores are not obliged to share a rate; each curve's fs/Nyquist is its own."""
    session = Session()
    fast = session.add_document(_doc(tmp_path, "fast.dat", sigma=QUIET, fs=FS, seed=1))
    slow = session.add_document(
        _doc(tmp_path, "slow.dat", sigma=QUIET, fs=FS / 5, n=N // 5, seed=2))
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(fast, 0)
    w.set_compare([slow])

    a, b = w.curves()
    assert a.fs == pytest.approx(FS)
    assert b.fs == pytest.approx(FS / 5)
    assert a.summary["nyquist_hz"] == pytest.approx(FS / 2)
    assert b.summary["nyquist_hz"] == pytest.approx(FS / 10)
    # Both rates are printed — neither curve is described with the other's time base.
    assert f"{FS:,.6g}" in w.details_text() and f"{FS / 5:,.6g}" in w.details_text()


def test_a_shorter_comparison_recording_is_clipped_and_says_so(qapp, qtbot, tmp_path):
    session = Session()
    long_id = session.add_document(_doc(tmp_path, "long.dat", sigma=QUIET, n=N, seed=1))
    short_id = session.add_document(_doc(tmp_path, "short.dat", sigma=QUIET, n=N // 4, seed=2))
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(long_id, 0)
    w.set_compare([short_id])

    assert len(w.curves()) == 2
    short = w.curves()[1]
    assert short.n_samples == N // 4
    assert "clipped" in short.note
    assert "clipped" in w.details_text()


# ================================================================================================
# 3. identity — a segment named after its parent
# ================================================================================================

def test_a_segment_is_offered_and_does_not_collide_with_its_parent(qapp, qtbot, tmp_path):
    """The persistent-key identity, in the PSD window: a crop named after the
    recording it came from must be its own row, not a shadow of the parent's."""
    session = Session()
    parent = session.add_document(_doc(tmp_path, "baseline.dat", sigma=QUIET, seed=1))
    seg = session.create_segment(parent, 0, N // 2, name="baseline.dat")   # same name!
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(parent, 0)

    # The segment is offered for comparison, under a label that is not the parent's.
    offered = w.comparable_labels()
    assert len(offered) == 1
    assert offered[0] != "baseline.dat", "identical labels would name nothing in a legend"

    w.set_compare([seg])
    assert len(w.curves()) == 2
    assert len({c.key for c in w.curves()}) == 2
    assert len({c.label for c in w.curves()}) == 2
    assert w.curves()[1].n_samples == N // 2


def test_switching_the_primary_moves_it_out_of_the_comparison_list(two_pores):
    """A recording cannot be compared with itself."""
    w, _quiet, _noisy = two_pores
    assert w.comparable_labels() == ["noisy.dat"]
    w._rec_combo.setCurrentIndex(1)                # the user picks the noisy pore
    assert w.comparable_labels() == ["quiet.dat"]
    assert len(w.curves()) == 1


# ================================================================================================
# 4. legend readability
# ================================================================================================

def test_the_legend_is_readable_on_both_backgrounds(two_pores):
    w, _quiet, noisy = two_pores
    w.set_compare([noisy])

    for mode, ink, canvas in (("dark", "#b9bfc7", "#111316"), ("light", "#1a1a1a", "#ffffff")):
        w.set_background(mode)
        legend = w.plot_item().legend
        assert legend is not None and legend.isVisible()
        assert legend.opts["labelTextColor"].name() == ink
        assert legend.opts["brush"].color().name() == canvas
        assert len(w.legend_labels()) >= 2


def test_no_legend_for_a_lone_curve(two_pores):
    w, _quiet, _noisy = two_pores
    legend = w.plot_item().legend
    assert legend is None or not legend.isVisible()


def test_the_legend_carries_each_curves_rms(two_pores):
    """The legend answers the question that was asked, not just "which line is which"."""
    w, _quiet, noisy = two_pores
    w.set_compare([noisy])
    labels = w.legend_labels()
    assert any("RMS" in t for t in labels)
    for curve in w.curves():
        entry = next(t for t in labels if t.startswith(curve.label))
        assert f"{curve.summary['total_rms']:.4g} {curve.units} RMS" in entry


def test_legend_entries_do_not_accumulate_across_redraws(two_pores):
    w, _quiet, noisy = two_pores
    w.set_compare([noisy])
    n_before = len(w.legend_labels())
    for _ in range(3):
        w.set_asd(not w.is_asd())
    assert len(w.legend_labels()) == n_before


# ================================================================================================
# 5. the overlay survives the display toggles
# ================================================================================================

@pytest.mark.parametrize("toggle", ["asd", "linear"])
def test_both_curves_follow_the_display_toggles(two_pores, toggle):
    w, _quiet, noisy = two_pores
    w.set_compare([noisy])
    psd_ys = [np.array(c.y) for c in w.curves()]

    if toggle == "asd":
        w.set_asd(True)
        for before, curve in zip(psd_ys, w.curves()):
            np.testing.assert_allclose(curve.y, np.sqrt(before), rtol=1e-6)
    else:
        w.set_log_log(False)
        assert all(c.item is not None for c in w.curves())
    assert len(w.curves()) == 2


# ================================================================================================
# 6. what is (and is not) comparable across two uncalibrated pores
# ================================================================================================

def test_absolute_magnitude_caution_appears_for_two_uncalibrated_pores(two_pores):
    """A PSD's y value is a squared CURRENT, so it carries the amplifier gain — and the
    gain here is uncalibrated. The shapes still compare; the heights do not, and the
    window says which is which rather than letting the user read the plot as if both did."""
    w, _quiet, noisy = two_pores
    w.set_compare([noisy])

    text = w.details_text()
    assert "CAUTION" in text
    assert "UNCALIBRATED" in text
    assert "SHAPE" in text, "the note must say what IS still comparable"


def test_the_caution_does_not_send_a_psd_user_to_frac_depth(two_pores):
    """REGRESSION. This window shares ``gains_comparable`` with the Plots window, and
    the shared default advice used to end *"dwell_s and frac_depth (dI/I0) ARE
    comparable — plot those instead."* — on a surface holding CALIBRATIONS and no
    detections, so it had no way to know whether those detections agree on a baseline
    window. ``frac_depth`` is ``depth_pA / |baseline_pA|``; on the owner's three-pore
    ladder, each probed to its own window, that sentence points at dI/I0 measured
    against three different baselines. The Plots window stopped saying it; this was
    the one caller the reconciliation did not reach.

    The caution keeps everything it can stand behind: the gain verdict, and the
    SHAPE of the spectrum, which is this surface's own honest alternative."""
    w, _quiet, noisy = two_pores
    w.set_compare([noisy])

    text = w.details_text()
    assert "CAUTION" in text
    assert "frac_depth" not in text and "dI/I0" not in text
    assert "dwell_s" not in text, "a noise window cannot vouch for an event column"
    assert "SHAPE" in text, "and it must still say what IS comparable"
    assert "comparable — plot" not in text, "no dangling advice clause"


def test_no_caution_for_a_lone_curve(two_pores):
    w, _quiet, _noisy = two_pores
    assert "CAUTION" not in w.details_text()


def test_a_shared_real_calibration_removes_the_caution(qapp, qtbot, tmp_path):
    from nanoscope.core.calibration import ChannelCalibration

    session = Session()
    a = session.add_document(_doc(tmp_path, "a.dat", sigma=QUIET, seed=1))
    b = session.add_document(_doc(tmp_path, "b.dat", sigma=NOISY, seed=2))
    cal = ChannelCalibration(gain=2.5, offset=0.0, target_units="pA")
    for doc_id in (a, b):
        session._calibration.set(doc_id, 0, cal)      # same real gain on both
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(a, 0)
    w.set_compare([b])

    assert len(w.curves()) == 2
    assert "CAUTION" not in w.details_text()


def test_mixed_units_are_called_out(qapp, qtbot, tmp_path):
    """A pA channel and an nA channel on one axis: the axis can only be labelled once."""
    session = Session()
    a = session.add_document(_doc(tmp_path, "pa.dat", sigma=QUIET, seed=1, units="pA"))
    b = session.add_document(_doc(tmp_path, "na.dat", sigma=QUIET, seed=2, units="nA"))
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(a, 0)
    w.set_compare([b])

    assert {c.units for c in w.curves()} == {"pA", "nA"}
    text = w.details_text()
    assert "DIFFERENT units" in text
    assert "pA" in text and "nA" in text
