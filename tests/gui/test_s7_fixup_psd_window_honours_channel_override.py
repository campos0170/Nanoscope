"""Sweep-7 fix-up X1: the PSD window lists the channel the app just calibrated and
detected on — it resolves the current channel where every other surface does.

Measured on the pre-fix-up tree (repro_X1_psd_window_ignores_override.py, on a COPY of
the 2 s undeclared CSV cut from the owner's -700 mV recording): after "Treat this channel
as: current" + pA + gain 2.5 through the real Calibration panel and a Detect that stored
15 events on ch0, the chip read "✓ Gain set: ×2.5 → pA", the rail "Events detected.", and
Analysis ▸ Power spectrum (noise / PSD)… was ENABLED — ``MainWindow._has_psd_recording``
asks the effective kind — yet the window it opened had an EMPTY Source combo and said
"No recording with a current channel and a known sample rate. Open one, then Refresh."
because ``PSDWindow._gather_recordings`` picked the first channel whose HEADER kind was
current. Sweep 7 moved the chip, rail, Detect gate, batch picker, Calibration panel and
Plots to ``Session.effective_kind`` and left this window on the header.

The fix (gui/psd_panel.py): ``_gather_recordings`` asks
``Session.resolve_current_channel(doc_id, None)`` — the one resolver — so the menu item
and the window it opens agree in both directions (a header-current channel re-declared
as voltage empties the window for the same reason it disables the menu). The window
computes on ``Session.calibrated`` already, so the spectrum it then draws is in pA at
the applied gain. Driven through a real MainWindow and the Calibration panel's own combo
and Apply button; the recording's path is under ``tmp_path``, never beside real data.
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
from nanoscope.gui.main_window import MainWindow
from nanoscope.gui.psd_panel import PSDWindow

FS = 50_000.0
N = 100_000
SIGMA = 1.0
GAIN = 2.5


def _doc(path, channels) -> TraceDocument:
    rng = np.random.default_rng(7)
    arrays = tuple(rng.normal(0.0, SIGMA, N).astype(np.float32) for _ in channels)
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=N, sample_rate_hz=FS,
                         rate_provenance=Provenance.FILE_META, channels=tuple(channels))
    return TraceDocument(meta=meta, channels=arrays)


def _undeclared(tmp_path) -> TraceDocument:
    """Neutral column headers, as a CSV or HDF5 declares them: both channels ``unknown``."""
    return _doc(tmp_path / "undeclared.csv", (
        ChannelInfo(index=0, name="Untitled", raw_units="", kind=ChannelKind.UNKNOWN),
        ChannelInfo(index=1, name="Untitled 1", raw_units="", kind=ChannelKind.UNKNOWN)))


def _declared(tmp_path) -> TraceDocument:
    return _doc(tmp_path / "declared.abf", (
        ChannelInfo(index=0, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE),
        ChannelInfo(index=1, name="IN15", raw_units="pA", kind=ChannelKind.CURRENT)))


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def _install(win: MainWindow, doc: TraceDocument) -> str:
    did = win._session.add_document(doc)
    win._create_default_layers(doc)
    return did


def _apply_treat_as_current(win: MainWindow, qtbot, did: str) -> None:
    """The user's gesture, through the panel's own widgets: Treat as current, pA, ×2.5, Apply."""
    win._on_selection(did, 0)
    cp = win._calibration_panel
    cp.set_selection(did, 0)
    qtbot.wait(20)
    cp._treat_as.setCurrentIndex(cp._treat_as.findData(ChannelKind.CURRENT))
    cp._units.clear()
    qtbot.keyClicks(cp._units, "pA")
    cp._gain.clear()
    qtbot.keyClicks(cp._gain, str(GAIN))
    cp._btn_apply.click()
    qtbot.wait(50)
    assert cp.status_text().startswith("Applied")
    assert win._session.effective_kind(did, 0) is ChannelKind.CURRENT


def test_the_psd_window_lists_a_channel_corrected_to_current_in_the_calibration_panel(
        window, qtbot, tmp_path):
    did = _install(window, _undeclared(tmp_path))
    window._on_selection(did, 0)
    window._sync_psd_action()
    assert not window._act_psd.isEnabled(), "precondition: nothing to spectrum yet"

    _apply_treat_as_current(window, qtbot, did)
    window._sync_psd_action()
    assert window._act_psd.isEnabled(), "the menu gate honours the override (sweep 7)"

    window._open_psd()
    ps = window._psd_window
    sources = [ps._rec_combo.itemText(i) for i in range(ps._rec_combo.count())]
    assert sources == ["undeclared.csv"], sources
    assert ps._current_target() == (did, 0)
    assert ps.has_data()
    assert "No recording with a current channel" not in ps.details_text()
    # The spectrum is of the CALIBRATED channel: pA at the applied gain.
    assert ps.units() == "pA"
    assert ps.summary()["total_rms"] == pytest.approx(GAIN * SIGMA, rel=0.05)


def test_an_already_open_psd_window_lists_the_recording_once_the_override_is_applied(
        window, qtbot, tmp_path):
    """The window open on another recording while the user corrects this one: the
    session's calibrationChanged re-points it, and the resolver it re-gathers through
    now agrees with the panel that did the correcting."""
    id_declared = _install(window, _declared(tmp_path))
    id_undeclared = _install(window, _undeclared(tmp_path))
    window._on_selection(id_declared, 1)
    window._open_psd()
    ps = window._psd_window
    assert [ps._rec_combo.itemText(i) for i in range(ps._rec_combo.count())] == ["declared.abf"]

    _apply_treat_as_current(window, qtbot, id_undeclared)
    qtbot.wait(50)
    sources = [ps._rec_combo.itemText(i) for i in range(ps._rec_combo.count())]
    assert sorted(sources) == ["declared.abf", "undeclared.csv"], sources
    assert ps._current_target() == (id_undeclared, 0)


def test_a_header_current_channel_re_declared_as_voltage_leaves_the_window_and_menu_agreeing(
        window, qtbot, tmp_path):
    """The reverse override: IN15 (header current) treated as voltage. The menu item goes
    dead AND the window lists nothing — the same answer from the same resolver."""
    did = _install(window, _declared(tmp_path))
    window._on_selection(did, 1)
    window._sync_psd_action()
    assert window._act_psd.isEnabled()
    window._open_psd()
    ps = window._psd_window
    assert ps._rec_combo.count() == 1

    window._session.set_calibration(
        did, 1, ChannelCalibration(1.0, 0.0, "pA", kind=ChannelKind.VOLTAGE),
        write_sidecar=False)
    qtbot.wait(50)
    window._sync_psd_action()
    assert not window._act_psd.isEnabled()
    ps.refresh()
    assert ps._rec_combo.count() == 0
    assert "No recording with a current channel" in ps.details_text()


def test_gather_recordings_is_the_session_resolver_not_the_header(qapp, qtbot, tmp_path):
    """The standalone window, no MainWindow: what it lists is what Session.resolve_current_channel
    returns, for a corrected-to-current channel and for one re-declared as voltage."""
    s = Session()
    did = s.add_document(_undeclared(tmp_path))
    w = PSDWindow(s)
    qtbot.addWidget(w)
    assert w._gather_recordings() == []
    s.set_calibration(did, 0, ChannelCalibration(GAIN, 0.0, "pA", kind=ChannelKind.CURRENT),
                      write_sidecar=False)
    recs = w._gather_recordings()
    assert [(r["doc_id"], r["channel"], r["channel_name"]) for r in recs] == [(did, 0, "Untitled")]
    assert recs[0]["channel"] == s.resolve_current_channel(did, None)

    s2 = Session()
    did2 = s2.add_document(_declared(tmp_path))
    w2 = PSDWindow(s2)
    qtbot.addWidget(w2)
    assert [(r["doc_id"], r["channel"]) for r in w2._gather_recordings()] == [(did2, 1)]
    s2.set_calibration(did2, 1, ChannelCalibration(1.0, 0.0, "pA", kind=ChannelKind.VOLTAGE),
                       write_sidecar=False)
    assert w2._gather_recordings() == []
