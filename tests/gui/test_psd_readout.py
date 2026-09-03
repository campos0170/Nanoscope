"""The PSD window's two read-outs: the log-log AXIS, and what α rests on.

Regressions for the apoferritin end-to-end run (report §7):

* **Medium-13 / High-7 (PSD half) — the axis.** On all three recordings the
  log-log frequency axis printed every minor tick of every decade and they ran
  together into ``4·10¹5·10⁶·10⁷·10⁸·10⁹·10¹00²``; the y axis mixed three number
  formats (``0.01, 0.1, 1, 10¹``) and carried an ``(x0.001)`` multiplier — on a
  LOG axis, where the tick numbers were consequently 1000x the ``pA²/Hz`` the
  label claimed. The tests here assert the three properties
  :class:`~nanoscope.gui.psd_panel._ReadableAxis` exists to hold: no two drawn
  labels touch, every label is in one notation, and **the number a label prints
  is the number at that position on the axis**.

* **Medium-14 — what α rests on.** A 0.4 s quiet segment leaves the fixed
  1-100 Hz flicker band holding exactly four FFT bins — the fewest the fit
  accepts — and the panel printed "α 1.207, R² 0.980" with nothing to say so.
  The bin count now travels with every exponent, a marginal fit says MARGINAL,
  an unavailable one says how many bins it found, and a window too short for the
  Welch parameters in force says that too.

Everything is synthetic; the real-recording proof (all three apoferritin files,
before/after) lives in the run's scratch evidence.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtGui

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.psd_panel import PSDWindow

FS = 50_000.0                 # the apoferritin recordings' rate
N_SAMPLES = 500_000           # 10 s
SIGMA = 3.0
PEDESTAL = -27.0              # the open-pore level these recordings sit at


# ================================================================================================
# builders
# ================================================================================================

def _doc(tmp_path, *, n: int = N_SAMPLES, name: str = "psd.dat") -> TraceDocument:
    """A current recording: white noise plus a 1/f-ish low-frequency component.

    The colouring matters only so the spectrum spans several decades in y, which is
    what makes the axis test a test of a REAL log-log plot rather than a flat line.
    """
    rng = np.random.default_rng(7)
    white = rng.normal(0.0, SIGMA, n)
    spec = np.fft.rfft(rng.normal(0.0, 1.0, n))
    f = np.fft.rfftfreq(n, d=1.0 / FS)
    shape = np.zeros_like(f)
    shape[1:] = f[1:] ** -0.5              # |H|² = 1/f
    pink = np.fft.irfft(spec * shape, n=n)
    pink *= SIGMA / (np.std(pink) or 1.0)
    sig = (white + pink + PEDESTAL).astype(np.float32)

    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


def _window(qtbot, tmp_path, *, n: int = N_SAMPLES) -> PSDWindow:
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, n=n))
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(doc_id, 0)
    w.resize(1500, 1000)          # a real, generous axis: crowding is not the excuse
    w.show()
    qtbot.waitExposed(w)
    return w


#: The −700 mV geometry: a 0.93 s quiet segment at 50 kHz, auto nperseg 4096, so the
#: spectrum runs 12.2 Hz - 25 kHz. THIS SPAN IS THE TEST. pyqtgraph's own per-level
#: crowding check does drop the minor labels once a log axis is wide enough (over ~4
#: decades), so a spectrum computed over a whole 10 s recording hides the defect
#: entirely; the smear the report screenshots is a ~3-decade axis, which is what a
#: quiet segment between events actually gives you.
_SMEAR_SEGMENT_S = (0.0, 46_590 / FS)


# ================================================================================================
# axis helpers — read what the axis would actually DRAW
# ================================================================================================

def _label_specs(axis) -> list[tuple[object, str]]:
    """``[(rect, text)]`` for every tick label this axis would paint."""
    pix = QtGui.QPixmap(8, 8)
    painter = QtGui.QPainter(pix)
    try:
        _axis_spec, _ticks, text_specs = axis.generateDrawSpecs(painter)
    finally:
        painter.end()
    return [(rect, text) for rect, _flags, text in text_specs]


_SUP_BACK = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def _label_value(text: str) -> float | None:
    """The number a ``10ⁿ`` / ``m·10ⁿ`` label denotes, or None if it is not one.

    Returning None is itself a finding: it means the axis used a second notation.
    """
    mantissa, _, power = text.rpartition("·")
    if not power.startswith("10"):
        return None
    exponent = power[2:].translate(_SUP_BACK)
    try:
        return float(mantissa or 1.0) * 10.0 ** int(exponent)
    except ValueError:
        return None


def _tick_pixel(axis, value: float) -> float:
    """Where ``value`` (data units) sits along ``axis``, in the axis's own pixels.

    Mirrors ``AxisItem.generateDrawSpecs``'s mapping so a label's printed number can
    be checked against the position it is printed at — the only test that catches an
    axis whose numbers are a silent multiple of the ones it is drawn over.
    """
    bounds = axis.mapRectFromParent(axis.geometry())
    v = np.log10(value) if axis.logMode else value
    lo, hi = axis.range
    dif = hi - lo
    if axis.orientation in ("left", "right"):
        scale = -bounds.height() / dif
        offset = lo * scale - bounds.height()
    else:
        scale = bounds.width() / dif
        offset = lo * scale
    return v * scale - offset


def _label_centre(rect, axis) -> float:
    return (rect.center().x() if axis.orientation in ("bottom", "top")
            else rect.center().y())


def _overlapping_pairs(axis) -> list[tuple[str, str]]:
    """Pairs of drawn labels whose rectangles intersect — i.e. read as one smear."""
    specs = _label_specs(axis)
    bad = []
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if specs[i][0].intersects(specs[j][0]):
                bad.append((specs[i][1], specs[j][1]))
    return bad


# ================================================================================================
# Medium-13 / High-7 — the axis
# ================================================================================================

@pytest.mark.parametrize("which", ["bottom", "left"])
def test_log_log_tick_labels_never_overlap(qapp, qtbot, tmp_path, which):
    """No two labels on a log-log axis may touch.

    BEFORE: the frequency axis drew 10¹ through 3·10⁴ — every minor tick of every
    decade — and 15-19 pairs of them intersected, on a 1055 px axis, on all three
    apoferritin recordings.
    """
    w = _window(qtbot, tmp_path)
    w.set_log_log(True)
    w.set_segment(*_SMEAR_SEGMENT_S)
    axis = w.plot_item().getAxis(which)

    assert len(_label_specs(axis)) >= 3          # the axis is labelled at all
    assert _overlapping_pairs(axis) == []


def _unthinned_labels(axis) -> list[str]:
    """What pyqtgraph itself would draw, BEFORE the panel's collision filter."""
    pix = QtGui.QPixmap(8, 8)
    painter = QtGui.QPainter(pix)
    try:
        _axis_spec, _ticks, text_specs = pg.AxisItem.generateDrawSpecs(axis, painter)
    finally:
        painter.end()
    return [text for _rect, _flags, text in text_specs]


@pytest.mark.parametrize("which", ["bottom", "left"])
def test_thinning_never_costs_a_decade(qapp, qtbot, tmp_path, which):
    """Whatever the filter drops, it may not drop a decade.

    A GUARD on the fix rather than a reproduction of the reported defect: nothing
    was thinned before, so this could not have failed then. It fails against the
    obvious wrong fix — thinning in emission order — which loses 10², 10³ and 10⁴
    on the −700 mV spectrum (proved by mutation in the run's scratch evidence).

    Dropping labels in the order they arrive would keep ``9·10¹`` and lose ``10²``:
    at this span pyqtgraph hands over ONE flat level of 31 ticks, so emission order
    carries no priority at all and the filter has to supply it. (Decades that
    pyqtgraph itself declines to draw — a label that would hang off the end of the
    axis — are its business, so the comparison is against its own candidate set.)
    """
    w = _window(qtbot, tmp_path)
    w.set_log_log(True)
    w.set_segment(*_SMEAR_SEGMENT_S)
    axis = w.plot_item().getAxis(which)

    drawn = {text for _rect, text in _label_specs(axis)}
    offered_decades = [t for t in _unthinned_labels(axis) if "·" not in t]
    # (the y axis clips the outermost decade label at the frame, so it offers two)
    assert len(offered_decades) >= 2, "the test data should span several decades"
    assert set(offered_decades) <= drawn, \
        f"decades lost to thinning: {sorted(set(offered_decades) - drawn)}"


@pytest.mark.parametrize("which", ["bottom", "left"])
def test_log_axis_uses_one_notation(qapp, qtbot, tmp_path, which):
    """One number format per axis.

    BEFORE: the y axis read ``0.01  0.1  1  10¹`` — plain decimals and an exponent
    on four ticks. Mixing them is what turns a decade error into an easy mistake.
    """
    w = _window(qtbot, tmp_path)
    w.set_log_log(True)
    w.set_segment(*_SMEAR_SEGMENT_S)
    axis = w.plot_item().getAxis(which)

    drawn = [text for _rect, text in _label_specs(axis)]
    assert drawn
    unparsed = [t for t in drawn if _label_value(t) is None]
    assert unparsed == [], f"labels not in the panel's 10ⁿ / m·10ⁿ notation: {unparsed}"


@pytest.mark.parametrize("which", ["bottom", "left"])
def test_axis_numbers_are_the_numbers_at_that_position(qapp, qtbot, tmp_path, which):
    """An axis may not print a number that is a multiple of the one it sits over.

    BEFORE: with the unit written into the label TEXT (``PSD (pA²/Hz)``) pyqtgraph
    had no unit to prefix, so it appended ``(x0.001)`` to the label and scaled every
    tick by 1000 — the shipped −700 mV spectrum's white floor of 0.002484 pA²/Hz was
    drawn against a tick reading between 1 and 10. This asserts the fix at the level
    the reader cares about: parse each label back to a number and check it lands
    where it is drawn.
    """
    w = _window(qtbot, tmp_path)
    w.set_log_log(True)
    w.set_segment(*_SMEAR_SEGMENT_S)
    axis = w.plot_item().getAxis(which)

    assert "(x" not in axis.labelString()          # no invented multiplier in the label
    assert axis.autoSIPrefixScale == 1.0

    for rect, text in _label_specs(axis):
        value = _label_value(text)
        assert value is not None, f"unreadable tick label {text!r}"
        assert _tick_pixel(axis, value) == pytest.approx(_label_centre(rect, axis), abs=1.5), \
            f"label {text!r} is not drawn at the position of the value it prints"


def test_axis_stays_readable_zoomed_inside_one_decade(qapp, qtbot, tmp_path):
    """Zoomed under a decade the axis must still be labelled AND still not overlap.

    Blanking every non-decade label (the sibling panel's rule) empties the axis here;
    dropping only what collides does not.
    """
    w = _window(qtbot, tmp_path)
    w.set_log_log(True)
    w.set_segment(*_SMEAR_SEGMENT_S)
    plot = w.plot_item()
    plot.setXRange(np.log10(200.0), np.log10(700.0), padding=0)
    qapp.processEvents()
    axis = plot.getAxis("bottom")

    assert len(_label_specs(axis)) >= 2
    assert _overlapping_pairs(axis) == []


# ================================================================================================
# Medium-14 — what α rests on
# ================================================================================================

def test_marginal_flicker_fit_reports_its_bin_count_and_says_marginal(qapp, qtbot, tmp_path):
    """A 0.4 s segment leaves the 1-100 Hz band holding exactly 4 bins.

    BEFORE: "1/f exponent α: 1.025 (magnitude at 1 Hz: 0.3439 pA²/Hz, R²=0.968)"
    and "fit over 24.41–100 Hz" — no count, no caution, an R² over four points
    reading exactly like an R² over four hundred.
    """
    w = _window(qtbot, tmp_path)
    w.set_segment(0.0, 0.3983)                   # 19,915 samples -> auto nperseg 2048
    assert w.segment_info()[0] == 2048

    fl = w.summary()["flicker"]
    assert fl["n"] == 4 and fl["status"] == "marginal"
    assert np.isfinite(fl["alpha"])              # still reported — but not bare

    text = w.details_text()
    assert "4 FFT bins" in text
    assert "MARGINAL" in text
    assert "CAUTION" in text
    assert f"{fl['alpha']:.3f}" in text          # the number itself is still there


def test_well_supported_flicker_fit_reports_its_bin_count_without_a_caution(
        qapp, qtbot, tmp_path):
    """The count travels with EVERY exponent — a good fit says what it rests on too."""
    w = _window(qtbot, tmp_path)
    w.set_segment(0.0, 5.0)
    fl = w.summary()["flicker"]
    assert fl["status"] == "ok" and fl["n"] >= 8

    text = w.details_text()
    assert f"on {fl['n']} FFT bins" in text
    assert "MARGINAL" not in text


def test_unavailable_flicker_says_how_many_bins_it_found(qapp, qtbot, tmp_path):
    """"Unavailable" must name the count, not just assert there were too few.

    BEFORE: "1/f exponent α:    unavailable (too few low-frequency bins)" — which
    never says how far short, so it cannot tell you how much longer a segment you
    need.
    """
    w = _window(qtbot, tmp_path)
    w.set_segment(0.0, 4096 / FS)                # 0.08 s -> auto nperseg 512, Δf ≈ 97.7 Hz
    fl = w.summary()["flicker"]
    assert fl["status"] == "unavailable" and not np.isfinite(fl["alpha"])
    assert fl["n_band"] < 4

    text = w.details_text()
    assert "unavailable" in text
    assert f"holds {fl['n_band']} FFT bin" in text
    assert str(4) in text                        # ... and how many it needs


def test_a_single_periodogram_says_it_is_one(qapp, qtbot, tmp_path):
    """nperseg over a short window silently stops Welch averaging.

    BEFORE: nperseg = 65536 on a 1.4 s segment printed "segments ≈ 1" among the
    parameters and nothing else; the α and R² that come out of a single
    periodogram look exactly like measurements.
    """
    w = _window(qtbot, tmp_path)
    w.set_segment(0.0, 1.4)
    w.set_nperseg(65536)
    assert w.segment_info()[2] == 1

    text = w.details_text()
    assert "single periodogram" in text
    assert "CAUTION" in text


def test_few_welch_averages_says_so(qapp, qtbot, tmp_path):
    """Under the automatic target of 8 averages the spectrum is noisy, and says so."""
    w = _window(qtbot, tmp_path)
    w.set_segment(0.0, 1.4)
    w.set_nperseg(16384)
    nseg = w.segment_info()[2]
    assert 1 < nseg < 8

    text = w.details_text()
    assert f"≈{nseg} Welch averages" in text


def test_the_overlay_legend_carries_the_bin_count_with_alpha(qapp, qtbot, tmp_path):
    """The fit line's legend entry is the one that gets cropped into a slide alone."""
    session = Session()
    a = session.add_document(_doc(tmp_path, name="a.dat"))
    b = session.add_document(_doc(tmp_path, name="b.dat"))
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(a, 0)
    w.resize(1500, 1000)
    w.show()
    qtbot.waitExposed(w)
    w.set_segment(0.0, 0.3983)
    w.set_compare([b])
    assert len(w.curves()) == 2

    fl = w.summary()["flicker"]
    entries = [str(label.text) for _item, label in w.plot_item().legend.items]
    fit_entry = next(e for e in entries if "1/f fit" in e)
    assert f"{fl['n']} bins" in fit_entry
    assert "MARGINAL" in fit_entry
