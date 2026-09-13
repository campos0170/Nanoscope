"""Specialist acquisition dialog and real project capture/restore seams."""
from dataclasses import replace

import numpy as np
import pytest

from nanoscope.core.acquisition import AcquisitionBandwidth
from nanoscope.core.acquisition_state import WORKSPACE_KEY
from nanoscope.core.document import TraceDocument
from nanoscope.core.project import load_project, save_project
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.acquisition_dialog import AcquisitionBandwidthDialog
from nanoscope.gui.main_window import MainWindow

HEADER = AcquisitionBandwidth(5000, "reported", "file_header",
                              header_fields=(("fSignalLowpassFilter", "5000"),))


def _doc(path, reverse=False):
    current = np.linspace(110, 130, 2000).astype(np.float32)
    voltage = np.full(current.size, -100, dtype=np.float32)
    infos = [ChannelInfo(0, "current", "pA", ChannelKind.CURRENT, acquisition_bandwidth=HEADER),
             ChannelInfo(1, "voltage", "mV", ChannelKind.VOLTAGE)]
    arrays = [current, voltage]
    if reverse:
        infos = [replace(info, index=i) for i, info in enumerate(reversed(infos))]
        arrays.reverse()
    return TraceDocument(RecordingMeta(path, "synthetic", len(current), 50000,
                                       Provenance.FILE_META, tuple(infos)), tuple(arrays))


@pytest.fixture
def dialog_rig(qapp, qtbot, tmp_path):
    path = tmp_path / "recording.dat"
    path.write_bytes(b"unchanged source fixture")
    session = Session()
    doc = _doc(path)
    key = session.add_document(doc)
    dialog = AcquisitionBandwidthDialog(session, key)
    qtbot.addWidget(dialog)
    return dialog, session, doc, key, path


def _reported(dialog, cutoff="2000"):
    dialog.mode.setCurrentIndex(dialog.mode.findData("reported"))
    dialog.cutoff.setText(cutoff)
    dialog.filter_type.setText("Bessel")
    dialog.order.setValue(8)
    dialog.note.setPlainText("Verified amplifier settings")


def test_dialog_applies_per_channel_override_without_touching_source(dialog_rig):
    dialog, session, doc, key, path = dialog_rig
    before = path.read_bytes()
    _reported(dialog)
    dialog.apply()
    assert session.acquisition_bandwidth(key, 0).cutoff_hz == 2000
    assert session.acquisition_bandwidth(key, 0).source == "user"
    assert doc.meta.channels[0].acquisition_bandwidth == HEADER
    assert "5000" in dialog.header.text()
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]
    dialog.channel.setCurrentIndex(1)
    assert dialog.mode.currentData() == "file"
    assert session.acquisition_bandwidth_override(key, 1) is None
    dialog.channel.setCurrentIndex(0)
    assert dialog.mode.currentData() == "reported"
    assert dialog.cutoff.text() == "2000"
    dialog.mode.setCurrentIndex(dialog.mode.findData("file"))
    dialog.apply()
    assert session.acquisition_bandwidth(key, 0) == HEADER


@pytest.mark.parametrize("cutoff", ["", " ", "0", "-1", "nan", "inf", "1e999", "not a number", "10 kHz", "1_20"])
def test_invalid_dialog_cutoff_is_refused_without_changing_metadata(dialog_rig, cutoff):
    dialog, session, doc, key, path = dialog_rig
    _reported(dialog, cutoff)
    dialog.apply()
    assert dialog.error.text() == (
        "Not applied: acquisition cutoff must be a finite positive number in Hz")
    assert session.acquisition_bandwidth_override(key, 0) is None
    assert doc.meta.channels[0].acquisition_bandwidth == HEADER
    assert path.read_bytes() == b"unchanged source fixture"


def test_unknown_and_bypassed_modes_do_not_infer_zero_bandwidth(dialog_rig):
    dialog, session, doc, key, path = dialog_rig
    for mode in ("unknown", "disabled"):
        dialog.mode.setCurrentIndex(dialog.mode.findData(mode))
        assert not dialog.cutoff.isEnabled()
        dialog.note.setPlainText("Only the selected amplifier stage is described")
        dialog.apply()
        value = session.acquisition_bandwidth(key, 0)
        assert value.status == mode
        assert value.cutoff_hz is None
        assert value.source == "user"


def test_loading_and_reapplying_preserves_precise_cutoff_and_large_order(dialog_rig):
    dialog, session, doc, key, path = dialog_rig
    value = AcquisitionBandwidth(2000.123456789, "reported", "user", "Bessel", 123,
                                 "Imported amplifier configuration")
    session.set_acquisition_bandwidth(key, 0, value)
    dialog._load()
    assert float(dialog.cutoff.text()) == value.cutoff_hz
    assert dialog.order.value() == 123
    dialog.apply()
    assert session.acquisition_bandwidth(key, 0) == value
    assert doc.meta.channels[0].acquisition_bandwidth == HEADER
    assert path.read_bytes() == b"unchanged source fixture"


def test_real_main_window_workspace_capture_and_restore_match_channel_identity(qapp, qtbot,
                                                                              tmp_path):
    source = MainWindow()
    qtbot.addWidget(source)
    path = tmp_path / "recording.dat"
    path.write_bytes(b"unchanged source fixture")
    doc = _doc(path)
    key = source._session.add_document(doc)
    source._create_default_layers(doc)
    source._on_selection(key, 0)
    source._dirty = False
    override = AcquisitionBandwidth(2000, "reported", "user", "Bessel", 8, "Confirmed")
    source._session.set_acquisition_bandwidth(key, 0, override)
    assert source._dirty
    state = source._gather_project_state()
    assert state.workspace.extra[WORKSPACE_KEY][0]["channel_name"] == "current"
    saved = save_project(tmp_path / "workflow.nanoscope", state)

    destination = MainWindow()
    qtbot.addWidget(destination)
    other_doc = _doc(path, reverse=True)
    other_key = destination._session.add_document(other_doc)
    destination._restore_state = load_project(saved)
    destination._apply_restore_for_doc(other_key)
    assert destination._session.acquisition_bandwidth(other_key, 1) == override
    assert destination._session.acquisition_bandwidth_override(other_key, 0) is None
    assert other_doc.meta.channels[1].acquisition_bandwidth == HEADER
    assert path.read_bytes() == b"unchanged source fixture"
    assert not (tmp_path / "recording.dat.npcal.json").exists()
    source._dirty = False
    destination._dirty = False
    destination._restore_state = None
