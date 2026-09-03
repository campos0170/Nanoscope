"""Sweep-7 fix-up X3: the File info panel's Kind column must read what a channel is TREATED
AS, not the header alone (docs/sweep7/FINDINGS.md, the X-combos skeptic).

Measured on a 2 s CSV of the owner's -700 mV recording with neutral column headers (both
channels ``unknown``), after "Treat this channel as: current" + pA + gain 2.5 was committed
through the real Apply button: the Calibration panel's title read
``Untitled · current (file says unknown)``, the Analysis panel's title the same, the chip
``✓ Gain set: ×2.5 → pA``, the rail offered DETECT -- and the File info panel's Channels
table read Kind = ``unknown`` for both channels, before and after the detection, with no
mention of the override. It is the panel a user opens to check what a channel IS, and it
was the one surface still printing the header alone (``file_info_panel.py:149``
``info.kind.value``).

The fix reads ``Session.effective_kind`` -- THE one resolver every other surface adopted in
sweep 7 -- and keeps the header's word in parentheses, the same shape as the two sibling
titles. Every test here reads the real table's cells; the last one drives the real
Calibration panel's combo, fields and Apply button on the same session so the two widgets
are compared as the user sees them.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.core.calibration import ChannelCalibration
from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.calibration_panel import CalibrationPanel
from nanoscope.gui.file_info_panel import FileInfoPanel

FS = 50_000.0
KIND_COL = 2


def _doc(path, channels, arrays) -> TraceDocument:
    n = int(arrays[0].size)
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=n, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=tuple(channels))
    return TraceDocument(meta=meta, channels=tuple(arrays))


def _undeclared(tmp_path) -> TraceDocument:
    """Two channels the file does not declare -- the shape the override was built for.
    ch0 is a -700 mV open pore in pA (raw median -49.13), ch1 its command voltage."""
    cur = np.full(40_000, -49.13, dtype=np.float32)
    volt = np.full(40_000, -700.0, dtype=np.float32)
    return _doc(tmp_path / "undeclared.csv", (
        ChannelInfo(index=0, name="Untitled", raw_units="", kind=ChannelKind.UNKNOWN),
        ChannelInfo(index=1, name="Untitled 1", raw_units="", kind=ChannelKind.UNKNOWN)),
        (cur, volt))


def _declared(tmp_path) -> TraceDocument:
    """A recording whose header declares its channels -- the owner's ABF shape."""
    cur = np.full(40_000, -57.68, dtype=np.float32)
    volt = np.full(40_000, -1.0, dtype=np.float32)
    return _doc(tmp_path / "declared.abf", (
        ChannelInfo(index=0, name="IN13", raw_units="V", kind=ChannelKind.RAW_VOLTAGE),
        ChannelInfo(index=1, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)),
        (volt, cur))


def _kinds(panel: FileInfoPanel) -> list[str]:
    return [panel._table.item(r, KIND_COL).text() for r in range(panel._table.rowCount())]


def _kind_tips(panel: FileInfoPanel) -> list[str]:
    return [panel._table.item(r, KIND_COL).toolTip() for r in range(panel._table.rowCount())]


@pytest.fixture
def info_panel(qapp, qtbot):
    s = Session()
    p = FileInfoPanel(s)
    qtbot.addWidget(p)
    p.show()
    return p, s


# ================================================================================================
# X3 -- the Kind column follows the override, with the header's word kept in parentheses
# ================================================================================================
def test_the_kind_column_reads_the_override_with_the_headers_word_in_parentheses(
        info_panel, tmp_path):
    panel, s = info_panel
    doc_id = s.add_document(_undeclared(tmp_path))
    panel.set_document(doc_id)
    assert _kinds(panel) == ["unknown", "unknown"]          # as loaded: the header, verbatim

    # The session door the Calibration panel's own Apply ends in (no sidecar: this is a
    # synthetic recording, and the rule is that nothing tests write lands beside data).
    s.set_calibration(doc_id, 0, ChannelCalibration(gain=2.5, target_units="pA",
                                                    kind=ChannelKind.CURRENT),
                      write_sidecar=False)

    assert _kinds(panel) == ["current (file says unknown)", "unknown"]
    tip0, tip1 = _kind_tips(panel)
    assert "override" in tip0 and "'unknown'" in tip0 and "'current'" in tip0
    assert tip1 == ""


def test_the_reverse_override_names_the_file_as_the_dissenter(info_panel, tmp_path):
    """IN15 (header current) re-declared a voltage: the cell says what it is treated as AND
    that the file disagrees -- the same sentence the workflow strip and the Detect gate use."""
    panel, s = info_panel
    doc_id = s.add_document(_declared(tmp_path))
    panel.set_document(doc_id)
    assert _kinds(panel) == ["raw_voltage", "current"]

    s.set_calibration(doc_id, 1, ChannelCalibration(kind=ChannelKind.VOLTAGE),
                      write_sidecar=False)

    assert _kinds(panel) == ["raw_voltage", "voltage (file says current)"]


def test_a_gain_edit_without_an_override_keeps_the_headers_word_verbatim(info_panel, tmp_path):
    """A plain gain on a declared channel is not an override: no parentheses, no tooltip --
    the panel's verbatim contract holds wherever the file and the user agree."""
    panel, s = info_panel
    doc_id = s.add_document(_declared(tmp_path))
    panel.set_document(doc_id)

    s.set_calibration(doc_id, 1, ChannelCalibration(gain=2.5, target_units="pA"),
                      write_sidecar=False)

    assert _kinds(panel) == ["raw_voltage", "current"]
    assert _kind_tips(panel) == ["", ""]


def test_clearing_the_override_returns_the_cell_to_the_headers_word(info_panel, tmp_path):
    panel, s = info_panel
    doc_id = s.add_document(_undeclared(tmp_path))
    panel.set_document(doc_id)
    s.set_calibration(doc_id, 0, ChannelCalibration(kind=ChannelKind.CURRENT),
                      write_sidecar=False)
    assert _kinds(panel)[0] == "current (file says unknown)"

    s.set_calibration(doc_id, 0, ChannelCalibration(), write_sidecar=False)

    assert _kinds(panel) == ["unknown", "unknown"]
    assert _kind_tips(panel) == ["", ""]


def test_the_kind_cell_agrees_with_the_calibration_panels_title_after_the_real_apply(
        qapp, qtbot, tmp_path):
    """The chain the skeptic drove: the real combo, the real fields, the real Apply button --
    and the two widgets read off the same session must say the same thing about ch0."""
    s = Session()
    (tmp_path / "undeclared.csv").write_bytes(b"x")     # the sidecar Apply writes lands here
    doc_id = s.add_document(_undeclared(tmp_path))
    fi = FileInfoPanel(s)
    cp = CalibrationPanel(s)
    for w in (fi, cp):
        qtbot.addWidget(w)
        w.show()
    fi.set_document(doc_id)
    cp.set_selection(doc_id, 0)
    assert _kinds(fi) == ["unknown", "unknown"]

    cp._treat_as.setCurrentIndex(cp._treat_as.findData(ChannelKind.CURRENT))
    for field, text in ((cp._units, "pA"), (cp._gain, "2.5")):
        field.clear()
        qtbot.keyClicks(field, text)
    cp._btn_apply.click()

    kinds = _kinds(fi)
    assert kinds == ["current (file says unknown)", "unknown"]
    assert kinds[0] in cp._title.text(), (kinds[0], cp._title.text())
