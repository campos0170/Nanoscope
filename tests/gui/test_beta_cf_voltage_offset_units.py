"""Beta regressions CF-1 + CF-5 — the Calibration panel's voltage read-outs.

CF-1: the command-voltage check and the open-pore conductance applied the voltage
channel's GAIN but not its OFFSET, so with a committed ``gain=0.1, offset=-50``
the stats table said -999.9 mV while "= effective" and "Applied voltage V" said
-949.9 mV on the same screen, and G = I₀/V was computed from the uncorrected
voltage. The invariant: the FULL calibration reaches every read-out that claims
to show the applied voltage — live-typed and committed paths both.

CF-5: the same two read-outs hard-formatted "... mV" for a voltage channel whose
raw units are UNDECLARED (the panel's own units row reads "raw = (unknown)") and
fed the conductance assuming mV. The current-units side was already guarded
("current units unknown -> conductance not in nS"); the voltage side must be
equally honest — say the units are unknown, refuse the nS claim.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.analysis import iv
from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
)
from nanoscope.gui.calibration_panel import CalibrationPanel


# ================================================================================================
# builders
# ================================================================================================

def _two_channel_doc(tmp_path, *, volt_raw: float, cur_pA: float,
                     volt_units: str = "mV", name: str = "two.dat") -> TraceDocument:
    """A voltage + current doc shaped like the real ABF/HDF5 recordings.

    ``volt_units=""`` is the HDF5 shape: the file never declares units, and the
    reader records that honestly (raw_units "" == NOT DECLARED, never a guess).
    """
    n = 4096
    volt = np.ascontiguousarray(np.full(n, volt_raw, dtype=np.float32))
    cur = np.ascontiguousarray(np.full(n, cur_pA, dtype=np.float32))
    v = ChannelInfo(index=0, name="IN14", raw_units=volt_units, kind=ChannelKind.VOLTAGE)
    c = ChannelInfo(index=1, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(
        path=tmp_path / name, format_name="synthetic", n_samples=n,
        sample_rate_hz=50_000.0, rate_provenance=Provenance.FILE_META, channels=(v, c))
    return TraceDocument(meta=meta, channels=(volt, cur))


def _value(text: str) -> float:
    """The leading number of a '<value> <units>' read-out label."""
    return float(text.split()[0])


@pytest.fixture
def cal_panel(qapp, qtbot):
    session = Session()
    panel = CalibrationPanel(session)
    qtbot.addWidget(panel)
    return panel, session


# ================================================================================================
# CF-1 — the offset reaches every applied-voltage read-out
# ================================================================================================

def test_committed_voltage_offset_reaches_effective_and_conductance(cal_panel, tmp_path):
    """Committed path: gain=0.1, offset=-50 on a +500-count DC-error channel.

    True command voltage is -9499 × 0.1 - 50 = -999.9 mV; every surface must say so."""
    panel, session = cal_panel
    doc = _two_channel_doc(tmp_path, volt_raw=-9499.0, cur_pA=-2000.0)
    doc_id = session.add_document(doc)
    session.set_calibration(doc_id, 0, ChannelCalibration(
        gain=0.1, offset=-50.0, target_units="mV", provenance=Provenance.UNKNOWN))
    panel.set_selection(doc_id, 0)

    true_v = -9499.0 * 0.1 - 50.0                               # -999.9 mV
    # The stats table and the command-voltage check may not disagree about the same fact.
    assert float(panel._stat_labels["median"][1].text()) == pytest.approx(true_v)
    assert _value(panel._recon_effective.text()) == pytest.approx(true_v)
    assert panel._recon_effective.text().endswith("mV")
    # ...and the conductance box uses the same corrected voltage, so G = I₀/V is right.
    assert _value(panel._cond_voltage.text()) == pytest.approx(true_v)
    assert _value(panel._cond_g.text()) == pytest.approx(
        iv.conductance_in_nS(-2000.0, "pA", true_v), rel=1e-3)


def test_typed_voltage_offset_updates_the_readouts_live(cal_panel, tmp_path):
    """Live path: the offset being dialed reaches the read-outs before any Apply."""
    panel, session = cal_panel
    doc = _two_channel_doc(tmp_path, volt_raw=-9499.0, cur_pA=-2000.0)
    doc_id = session.add_document(doc)
    panel.set_selection(doc_id, 0)                              # defaults: gain 1, offset 0

    panel._gain.setText("0.1")
    panel._offset.setText("-50")
    true_v = -9499.0 * 0.1 - 50.0
    assert _value(panel._recon_effective.text()) == pytest.approx(true_v)
    assert _value(panel._cond_voltage.text()) == pytest.approx(true_v)
    assert _value(panel._cond_g.text()) == pytest.approx(
        iv.conductance_in_nS(-2000.0, "pA", true_v), rel=1e-3)


def test_sibling_voltage_commit_reaches_the_conductance_while_current_selected(cal_panel,
                                                                               tmp_path):
    """The current channel is selected; committing the VOLTAGE channel's offset must still
    refresh 'Applied voltage V' and G — the sibling-refresh path of calibrationChanged."""
    panel, session = cal_panel
    doc = _two_channel_doc(tmp_path, volt_raw=-9499.0, cur_pA=-2000.0)
    doc_id = session.add_document(doc)
    panel.set_selection(doc_id, 1)                              # the CURRENT channel

    session.set_calibration(doc_id, 0, ChannelCalibration(
        gain=0.1, offset=-50.0, target_units="mV", provenance=Provenance.UNKNOWN))
    true_v = -9499.0 * 0.1 - 50.0
    assert _value(panel._cond_voltage.text()) == pytest.approx(true_v)
    assert _value(panel._cond_g.text()) == pytest.approx(
        iv.conductance_in_nS(-2000.0, "pA", true_v), rel=1e-3)


def test_reconciliation_arithmetic_shown_matches_the_effective(cal_panel, tmp_path):
    """The '× gain + offset' row states the full transform, so the box's own arithmetic
    (measured × gain + offset = effective) holds on screen with a nonzero offset."""
    panel, session = cal_panel
    doc = _two_channel_doc(tmp_path, volt_raw=-9499.0, cur_pA=-2000.0)
    doc_id = session.add_document(doc)
    panel.set_selection(doc_id, 0)
    panel._gain.setText("0.1")
    panel._offset.setText("-50")

    measured = float(panel._recon_measured.text())
    gain_txt = panel._recon_gain.text()                         # "0.1 + -50"
    gain, offset = (float(p) for p in gain_txt.split("+"))
    assert measured * gain + offset == pytest.approx(_value(panel._recon_effective.text()))


# ================================================================================================
# CF-5 — unknown voltage units are never asserted to be mV
# ================================================================================================

def test_unknown_voltage_units_are_not_labelled_mV(cal_panel, tmp_path):
    """HDF5 shape: no units declared anywhere. The read-outs say so instead of 'mV'."""
    panel, session = cal_panel
    doc = _two_channel_doc(tmp_path, volt_raw=0.52, cur_pA=-100.0, volt_units="")
    doc_id = session.add_document(doc)

    panel.set_selection(doc_id, 0)                              # the unknown-units voltage channel
    assert "(unknown)" in panel._units_row.text()
    assert not panel._recon_effective.text().endswith("mV")
    assert "units unknown" in panel._recon_effective.text()

    panel.set_selection(doc_id, 1)                              # the current channel
    assert not panel._cond_voltage.text().endswith("mV")
    assert "units unknown" in panel._cond_voltage.text()


def test_unknown_voltage_units_refuse_the_nS_conductance_with_the_reason(cal_panel, tmp_path):
    """Symmetric with the current-units guard: no fabricated mV, no fabricated nS, and the
    note names the voltage units as the reason."""
    panel, session = cal_panel
    doc = _two_channel_doc(tmp_path, volt_raw=0.52, cur_pA=-100.0, volt_units="")
    doc_id = session.add_document(doc)
    panel.set_selection(doc_id, 1)

    assert panel._cond_g.text() == "—"
    assert "voltage units (unknown) -> conductance not in nS" in panel._cond_note.text()


def test_setting_the_voltage_units_restores_mV_and_the_conductance(cal_panel, tmp_path):
    """The refusal is the unknown units, nothing else: declare them (the guide's own
    instruction for HDF5 — set units in Calibration) and mV + nS come back."""
    panel, session = cal_panel
    doc = _two_channel_doc(tmp_path, volt_raw=-500.0, cur_pA=-100.0, volt_units="")
    doc_id = session.add_document(doc)
    panel.set_selection(doc_id, 0)

    panel._units.setText("mV")                                  # typed live, no Apply needed
    assert panel._recon_effective.text().endswith("mV")
    assert _value(panel._cond_voltage.text()) == pytest.approx(-500.0)
    assert _value(panel._cond_g.text()) == pytest.approx(
        iv.conductance_in_nS(-100.0, "pA", -500.0), rel=1e-3)


def test_volts_declared_voltage_scales_the_conductance_correctly(cal_panel, tmp_path):
    """A voltage channel in V must not be divided as if it were mV: pA/V -> nS is ×1e-3,
    exactly the to_nS table — the fabricated-mV bug made this 1000x too large."""
    panel, session = cal_panel
    doc = _two_channel_doc(tmp_path, volt_raw=0.5, cur_pA=-100.0, volt_units="V")
    doc_id = session.add_document(doc)
    panel.set_selection(doc_id, 1)

    assert panel._cond_voltage.text().endswith("V")
    assert not panel._cond_voltage.text().endswith("mV")
    # -100 pA / 0.5 V = -200 pA/V -> -0.2 nS (NOT -200 nS).
    assert _value(panel._cond_g.text()) == pytest.approx(
        iv.to_nS(-100.0 / 0.5, "pA", "V"), rel=1e-3)
    assert _value(panel._cond_g.text()) == pytest.approx(-0.2, rel=1e-3)
