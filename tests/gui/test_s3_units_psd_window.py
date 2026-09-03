"""S3 units-honesty on the Power-spectrum window: a spectrum reports the units it
was given and never supplies one of its own (JM-06).

``PSDWindow`` built its curve with ``units=chan.units or "pA"``. For an HDF5 —
which declares no units at all (io/hdf5.py leaves ``raw_units=""`` on every
channel and attaches "Units are not declared in this file. Magnitudes suggest
nA / V / nS"; docs/USER_GUIDE.md line 88 says the user sets them in Calibration)
— the "RMS in band" read-out therefore printed "0.004204 pA": a unit the file
never stated, and a magnitude 1000x off if the loader's own guess is right. 4.2 fA
RMS is below any patch-clamp amplifier's noise floor, so the first QC number a
nanopore user reads described an impossibly quiet pore.

This is the same guard CF-5 put on the Calibration panel and EXPL2-5/JHT-02 put on
the I-V window, in the same voice ("(units unknown)"), applied at this window's
seam: the curve carries ``chan.units`` verbatim, and every surface that prints it
— the band read-out, the details block, the legend, the axis label, the
mixed-units caution — names the undeclared state instead of filling the blank.

A declared unit must still be reported exactly as before; those cases are pinned
here too, because a guard that also breaks pA would be no better.
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
N = 100_000            # 2 s
PEDESTAL = 100.0
SIGMA = 4.0


def _doc(tmp_path, name: str, *, units: str = "", sigma: float = SIGMA,
         seed: int = 0) -> TraceDocument:
    """A current recording whose units default to ``""`` — the HDF5/CSV state."""
    rng = np.random.default_rng(seed)
    sig = (rng.normal(0.0, sigma, N) + PEDESTAL).astype(np.float32)
    info = ChannelInfo(index=0, name="current", raw_units=units, kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
        rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


def _window(qtbot, session: Session, doc_id: str) -> PSDWindow:
    w = PSDWindow(session)
    qtbot.addWidget(w)
    w.set_target(doc_id, 0)
    return w


@pytest.fixture
def undeclared(qapp, qtbot, tmp_path):
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, "pore.h5"))
    return _window(qtbot, session, doc_id)


# ================================================================================================
# the invented unit is gone — everywhere the window prints one
# ================================================================================================

def test_the_curve_carries_the_channels_own_units(undeclared):
    """No default: the channel declared nothing, so the curve says nothing."""
    (curve,) = undeclared.curves()
    assert curve.units == ""
    assert undeclared.units() == ""


def test_the_band_rms_readout_never_says_pA(undeclared):
    """The repro's surface: the QC number a nanopore user reads first."""
    text = undeclared.rms_readout_text()
    assert "pA" not in text
    assert "(units unknown)" in text
    # The VALUE is unchanged — this is a labelling guard, not a new measurement.
    assert undeclared.rms_value() == pytest.approx(SIGMA, rel=0.1)


def test_the_details_block_names_the_undeclared_state_and_the_fix(undeclared):
    text = undeclared.details_text()
    assert "pA" not in text
    assert "Total RMS noise:" in text and "(units unknown)" in text
    assert "(units unknown)²/Hz" in text            # the white floor keeps its /Hz
    # Said once, with the instruction that actually fixes it.
    assert "declares no units" in text
    assert "Calibration" in text


def test_the_axis_label_names_it_too(undeclared):
    label = undeclared.plot_item().getAxis("left").labelText
    assert "pA" not in label
    assert "units unknown" in label
    undeclared.set_asd(True)
    assert "pA" not in undeclared.plot_item().getAxis("left").labelText


def test_a_declared_unit_is_still_reported_exactly(qapp, qtbot, tmp_path):
    """The guard must not cost the ordinary case: a pA channel is still pA."""
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, "pore.abf", units="pA"))
    w = _window(qtbot, session, doc_id)

    assert w.units() == "pA"
    assert "pA" in w.rms_readout_text()
    assert "(units unknown)" not in w.rms_readout_text()
    details = w.details_text()
    assert "pA²/Hz" in details
    assert "declares no units" not in details
    assert "pA" in w.plot_item().getAxis("left").labelText


# ================================================================================================
# the overlay: an undeclared curve is named, not silently merged into a labelled one
# ================================================================================================

def test_an_overlay_mixing_declared_and_undeclared_units_says_so(qapp, qtbot, tmp_path):
    """Two curves, one of them unit-less: the caution has to name both states, and
    the legend entry must not borrow the other curve's unit."""
    session = Session()
    known = session.add_document(_doc(tmp_path, "known.abf", units="pA", seed=1))
    blank = session.add_document(_doc(tmp_path, "blank.h5", seed=2))
    w = _window(qtbot, session, known)
    w.set_compare([blank])

    assert {c.units for c in w.curves()} == {"pA", ""}
    text = w.details_text()
    assert "DIFFERENT units" in text
    assert "(units unknown)" in text                 # the blank one is NAMED in the caution
    legend = " | ".join(w.legend_labels())
    assert "blank.h5" in legend
    # "blank.h5 (4.02 pA RMS)" would be the old fabrication, on the other pore's unit.
    assert "blank.h5  (" in legend and "pA RMS)" not in legend.split("blank.h5")[1]
