"""C4 + C8 through the widgets: the app must never PRINT a number it does not have.

The pure half of this invariant is ``tests/analysis/test_nonfinite_guard.py``. This file
drives the surfaces the user actually reads, because the number on screen is the whole
defect: a recording with a single blank cell showed **"Total RMS noise: 0 pA"**, which is
what a flawless pore looks like.

Also here: the Gain box. ``QDoubleValidator`` accepts ``1e999`` and ``float()`` returns
``inf`` — the same defect as ``NaN`` in a sidecar, arriving through a different door, and
stopped by the same check on ``ChannelCalibration``.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os
import warnings

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from nanoscope.core.document import TraceDocument
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.calibration_panel import CalibrationPanel
from nanoscope.gui.file_info_panel import FileInfoPanel
from nanoscope.gui.psd_panel import PSDWindow

FS = 50_000.0
N = 200_000
SIGMA = 3.0
PEDESTAL = 100.0


# ================================================================================================
# builders
# ================================================================================================
def _doc(tmp_path, *, bad_at: int | None = None, name="rec.dat") -> TraceDocument:
    rng = np.random.default_rng(0)
    sig = (rng.normal(0.0, SIGMA, N) + PEDESTAL).astype(np.float32)
    if bad_at is not None:
        sig[bad_at] = np.nan
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / name, format_name="synthetic", n_samples=N,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(sig,))


def _psd_window(qtbot, tmp_path, *, bad_at=None):
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, bad_at=bad_at))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # the source warning is asserted elsewhere
        w = PSDWindow(session)
        qtbot.addWidget(w)
        w.set_target(doc_id, 0)
    return w, session, doc_id


# ================================================================================================
# C4 — the read-out
# ================================================================================================
def test_c4_the_noise_panel_does_not_print_zero_for_a_poisoned_recording(qapp, qtbot, tmp_path):
    """THE regression, on the surface where the number was read.

    Pre-fix the whole chain produced "Total RMS noise: 0 pA" for a trace whose true σ is
    3 pA. The read-out is fed from ``PSDCurve.summary`` / ``PSDCurve.band_rms``, so that is
    what this drives — a curve built exactly as ``_compute_curve`` builds it, from a
    recording carrying one NaN.
    """
    from nanoscope.analysis import psd as psd_mod
    from nanoscope.gui.psd_panel import PSDCurve

    sig = np.asarray(_doc(tmp_path, bad_at=12_345).raw(0), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        freqs, psd = psd_mod.welch_psd(sig, FS)
        summary = psd_mod.noise_summary(freqs, psd, FS)

    curve = PSDCurve(key="k", doc_id="d", channel=0, label="rec.dat", colour="#fff",
                     freqs=freqs, psd=psd, summary=summary, units="pA", fs=FS)
    w, _, _ = _psd_window(qtbot, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        details = w._curve_details(curve, header=False)

    assert "0 pA" not in details, "the panel reported zero noise for a poisoned recording"
    assert "Total RMS noise:   unavailable" in details
    assert "not finite" in details            # ...and says why


def test_c4_the_band_rms_readout_is_unavailable_not_zero(qapp, qtbot, tmp_path):
    """The live band read-out, same surface, same rule."""
    from nanoscope.analysis import psd as psd_mod
    from nanoscope.gui.psd_panel import PSDCurve

    sig = np.asarray(_doc(tmp_path, bad_at=12_345).raw(0), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        freqs, psd = psd_mod.welch_psd(sig, FS)
        curve = PSDCurve(key="k", doc_id="d", channel=0, label="rec.dat", colour="#fff",
                         freqs=freqs, psd=psd,
                         summary=psd_mod.noise_summary(freqs, psd, FS), units="pA", fs=FS)
        w, _, _ = _psd_window(qtbot, tmp_path)
        w._curves = [curve]
        w._update_rms_readout()

    text = w._rms_label.text()
    assert "unavailable" in text
    assert not text.startswith("0 ")


def test_c4_the_plot_names_the_real_reason_it_is_empty(qapp, qtbot, tmp_path):
    """An all-NaN spectrum has nothing to draw on ANY axis. Blaming the log-log toggle sent
    the user to fiddle with an axis setting instead of looking at their recording."""
    w, _, _ = _psd_window(qtbot, tmp_path, bad_at=12_345)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        details = w.details_text()

    assert "log-log" not in details
    assert "not finite" in details
    assert "File info" in details and "Issues" in details


def test_a_clean_recording_still_reports_its_sigma(qapp, qtbot, tmp_path):
    """The honesty fix must not cost the correct answer on good data.

    Asserted line by line, not against the whole details blob: the flicker section says
    "unavailable (too few low-frequency bins)" for its own unrelated reason, and a blanket
    ``"unavailable" not in details`` would be a test that passes by luck of the fixture.
    """
    w, _, _ = _psd_window(qtbot, tmp_path)
    assert w.summary()["total_rms"] == pytest.approx(SIGMA, rel=0.05)

    lines = w.details_text().splitlines()
    rms = next(ln for ln in lines if ln.startswith("Total RMS noise:"))
    assert "unavailable" not in rms and "pA" in rms
    assert f"{SIGMA:.4g}"[:3] in rms

    bands = [ln for ln in lines if ln.startswith("  <1 kHz") or ln.startswith("  1-10 kHz")]
    assert bands and all("unavailable" not in ln and "pA" in ln for ln in bands)
    # ...and no "the spectrum is not finite" explanation on a perfectly good recording.
    assert "not finite" not in w.details_text()


def test_c4_the_file_info_panel_shows_the_load_issue(qapp, qtbot, tmp_path):
    """The issues box is where a user finds out WHY the noise read-out gave up."""
    session = Session()
    doc_id = session.add_document(_doc(tmp_path, bad_at=999))
    panel = FileInfoPanel(session)
    qtbot.addWidget(panel)
    panel.set_document(doc_id)

    text = panel._issues.toPlainText()
    assert "non-finite" in text
    assert text != "(none)"


def test_the_file_info_panel_says_none_for_a_clean_recording(qapp, qtbot, tmp_path):
    session = Session()
    doc_id = session.add_document(_doc(tmp_path))
    panel = FileInfoPanel(session)
    qtbot.addWidget(panel)
    panel.set_document(doc_id)
    assert panel._issues.toPlainText() == "(none)"


# ================================================================================================
# C8 — the Gain box
# ================================================================================================
def _cal_panel(qtbot, tmp_path):
    session = Session()
    (tmp_path / "rec.dat").write_bytes(b"x")
    doc_id = session.add_document(_doc(tmp_path))
    panel = CalibrationPanel(session)
    qtbot.addWidget(panel)
    panel._confirm_sidecar_removal = lambda path, **kw: True
    panel.set_selection(doc_id, 0)
    return panel, session, doc_id


def test_c8_a_gain_of_1e999_cannot_be_committed(qapp, qtbot, tmp_path):
    """``QDoubleValidator`` accepts it (well-formed scientific notation, unbounded range) and
    ``float("1e999")`` is ``inf``. Committing it would make every sample of the recording
    infinite — the identical defect as a NaN in a sidecar, through the Gain box."""
    panel, session, doc_id = _cal_panel(qtbot, tmp_path)
    before = session.calibration(doc_id, 0)

    panel._gain.setText("1e999")
    panel._offset.setText("0")
    panel._apply()

    assert session.calibration(doc_id, 0) == before, "an infinite gain was committed"


def test_c8_the_preview_declines_an_infinite_gain_instead_of_raising(qapp, qtbot, tmp_path):
    panel, _, _ = _cal_panel(qtbot, tmp_path)
    panel._gain.setText("1e999")
    assert panel._pending() is None
    # Strengthened: the preview must name WHY, not fall back to the generic "enter a
    # number" it prints for `abc`. "Locked is styling, not setEnabled(False)" applies to a
    # refusal message too — a form that declines has to say what it declined.
    text = panel._formula.text()
    assert "finite" in text and "infinite one makes every sample infinite" in text, text


def test_a_normal_gain_still_commits(qapp, qtbot, tmp_path):
    panel, session, doc_id = _cal_panel(qtbot, tmp_path)
    panel._gain.setText("0.25")
    panel._offset.setText("1.5")
    panel._apply()
    cal = session.calibration(doc_id, 0)
    assert (cal.gain, cal.offset) == (0.25, 1.5)


# ================================================================================================
# C8 — a damaged project must not abort the restore
# ================================================================================================
def test_a_project_holding_a_non_finite_calibration_still_reopens(qapp, qtbot, tmp_path):
    """A project is a file on disk and can be damaged. The channel keeps its default and the
    restore reports it, rather than the whole reopen dying on one bad number.

    NOTE for anyone re-running this against restored pre-fix behaviour: this one HANGS
    there rather than failing, so the pre-fix verification run deselects it. The other
    six cases in this file fail cleanly.
    """
    from nanoscope.core.project import CalibrationRec, DocumentRef, ProjectState
    from nanoscope.core.segment import persistent_key
    from nanoscope.gui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    session = win._session
    (tmp_path / "rec.dat").write_bytes(b"x")
    doc_id = session.add_document(_doc(tmp_path))

    # Keyed by the PERSISTENT key (the source path), which is what a project stores — never
    # the per-load doc_id, which is this session's alone (core/identity.py).
    key = persistent_key(session.document(doc_id))
    win._restore_state = ProjectState(
        documents=[DocumentRef(path=key)],
        calibrations=[
            CalibrationRec(doc_id=key, channel=0, gain=float("inf"), offset=0.0,
                           target_units="pA", provenance="unknown"),
        ],
    )
    win._restore_bad_calibrations = []
    win._apply_restore_for_doc(doc_id)

    assert session.calibration(doc_id, 0).gain == 1.0          # kept its seeded default
    assert win._restore_bad_calibrations, "the restore swallowed a damaged calibration"
