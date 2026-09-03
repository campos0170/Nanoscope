"""§2.5 / §2.8 / §2.10 through the real widgets — the measured failures, end to end.

The unit-level rules live in ``tests/core/test_persist_invariant.py``. These are the same
three failures driven the way the user meets them, because each of them was invisible
precisely at that level:

* **§2.5** the note is not lost by the restore; it is lost by the ORDINARY APPLY that
  follows it, which faithfully carries forward the emptiness the restore installed. Nothing
  short of "open a project, change a gain, press Apply" reproduces it.
* **§2.8** ``gain = 0`` printed *"Total RMS noise: 0 pA"* with ``meta.issues = ()`` — the
  most convincing-looking QC result a noise panel can produce — and the panel's own guidance
  sent the user to a box that said "(none)".
* **§2.10** the disclosure of a partial export was a status line that clears after twelve
  seconds.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from conftest import live_doc_id
from nanoscope.core.calibration import ChannelCalibration, sidecar_path
from nanoscope.core.document import TraceDocument
from nanoscope.core.project import (
    CalibrationRec,
    DocumentRef,
    ProjectState,
    load_project,
    save_project,
)
from nanoscope.core.session import Session
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui.calibration_panel import CalibrationPanel
from nanoscope.gui.file_info_panel import FileInfoPanel
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 40_000
NOTE = "electrode replaced 2026-07-01 — gain measured against the 1 MΩ standard"


# ================================================================================================
# builders
# ================================================================================================
def _trace(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in [(4_000, 4_300), (12_000, 12_250), (25_000, 25_400)]:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _write_csv(path: Path, arr: np.ndarray) -> None:
    t = np.arange(arr.size, dtype=np.float64) / FS
    pd.DataFrame({"time": t, "current": np.asarray(arr, dtype=np.float64)}).to_csv(
        path, index=False)


def _doc(path: Path, arr: np.ndarray) -> TraceDocument:
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=arr.size,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(info,))
    return TraceDocument(meta=meta, channels=(arr,))


def _write_sidecar(csv: Path, *, gain: float, note: str) -> Path:
    sc = sidecar_path(csv)
    sc.write_text(json.dumps({"version": 1, "channels": {
        "0": {"gain": gain, "offset": 0.0, "target_units": "pA", "note": note}}}, indent=2))
    return sc


def _note_on_disk(sc: Path, channel: str = "0") -> str:
    return json.loads(sc.read_text())["channels"][channel]["note"]


def _window(qtbot) -> MainWindow:
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _panel_for(qtbot, session: Session, doc_id: str, ch: int = 0) -> CalibrationPanel:
    p = CalibrationPanel(session)
    qtbot.addWidget(p)
    p._confirm_sidecar_removal = lambda path, **kw: True
    p.set_selection(doc_id, ch)
    return p


def _downgrade_project_to_v4(proj: Path) -> None:
    """Rewrite a saved project's manifest as a v4 one: no ``note`` key anywhere.

    A genuine pre-v5 file, not a v5 file with the field blanked — the whole point of the
    three-state rule is that those two are different, and a test that could not build the
    first could not pin it.
    """
    with zipfile.ZipFile(proj) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    manifest = json.loads(members["manifest.json"])
    manifest["version"] = 4
    for rec in manifest["calibrations"]:
        rec.pop("note", None)
    members["manifest.json"] = json.dumps(manifest, indent=2).encode()
    with zipfile.ZipFile(proj, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in members.items():
            zf.writestr(name, blob)


# ================================================================================================
# §2.5 — the note the project could not carry
# ================================================================================================
def test_the_ordinary_apply_after_a_restore_keeps_the_sidecar_note(qapp, qtbot, tmp_path):
    """THE measured failure.

    Open a project, change a gain 0.5 → 0.75, press Apply: the note became ``""`` with
    **zero prompts** and a status line reading "Applied · saved to …". The sidecar
    confirmation never fires because this is a WRITE, not the removal it watches for — and
    the note is the one field a project file could not round-trip, which is the entire
    reason that confirmation exists.
    """
    csv = tmp_path / "pore.csv"
    _write_csv(csv, _trace())
    sidecar = _write_sidecar(csv, gain=0.5, note=NOTE)

    # A project saved BY THIS BUILD from a session that has the note live.
    proj = tmp_path / "p.nanoscope"
    save_project(proj, ProjectState(
        documents=[DocumentRef(path=str(csv.resolve()))],
        calibrations=[CalibrationRec.from_calibration(
            str(csv.resolve()), 0,
            ChannelCalibration(gain=0.5, offset=0.0, target_units="pA", note=NOTE))]))

    w = _window(qtbot)
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(str(proj))
    qtbot.wait(50)
    live = live_doc_id(w._session, csv)
    assert live is not None
    assert w._session.calibration(live, 0).note == NOTE, "the restore itself dropped the note"

    panel = _panel_for(qtbot, w._session, live)
    prompts: list = []
    panel._confirm_sidecar_removal = lambda path, **kw: (prompts.append(path), True)[1]
    panel._gain.setText("0.75")
    panel._apply()

    assert _note_on_disk(sidecar) == NOTE, "an ordinary Apply blanked the user's note"
    assert json.loads(sidecar.read_text())["channels"]["0"]["gain"] == pytest.approx(0.75)
    assert prompts == []
    w._controller.abort()


def test_a_pre_v5_project_leaves_the_sidecars_note_alone(qapp, qtbot, tmp_path):
    """The user's EXISTING projects are v3/v4 and carry no note at all.

    A record that says nothing must not be read as saying "there is none" — that reading is
    the defect. The sidecar was loaded moments earlier by ``Session.add_document``, and it is
    the only thing on the machine that knows.
    """
    csv = tmp_path / "pore.csv"
    _write_csv(csv, _trace())
    sidecar = _write_sidecar(csv, gain=0.5, note=NOTE)

    proj = tmp_path / "old.nanoscope"
    save_project(proj, ProjectState(
        documents=[DocumentRef(path=str(csv.resolve()))],
        calibrations=[CalibrationRec(doc_id=str(csv.resolve()), channel=0, gain=0.75,
                                     offset=0.0, target_units="pA", provenance="unknown")]))
    _downgrade_project_to_v4(proj)
    assert load_project(proj).calibrations[0].note is None

    w = _window(qtbot)
    with qtbot.waitSignal(w.projectRestored, timeout=30_000):
        w._open_project(str(proj))
    qtbot.wait(50)
    live = live_doc_id(w._session, csv)
    assert w._session.calibration(live, 0).gain == pytest.approx(0.75), "the project lost"
    assert w._session.calibration(live, 0).note == NOTE

    panel = _panel_for(qtbot, w._session, live)
    panel._gain.setText("0.9")
    panel._apply()
    assert _note_on_disk(sidecar) == NOTE
    w._controller.abort()


def test_a_project_saved_now_carries_the_note_itself(qapp, qtbot, tmp_path):
    """The other half: a project that carries the note restores it even with NO sidecar —
    a segment has no sidecar at all, and neither does a recording whose sidecar was deleted.
    """
    csv = tmp_path / "pore.csv"
    _write_csv(csv, _trace())
    _write_sidecar(csv, gain=0.5, note=NOTE)

    w = _window(qtbot)
    w.open_paths([str(csv)])
    # Drain the open pipeline rather than waiting a fixed interval: the load is
    # asynchronous, so a fixed wait can report success before anything happened.
    qtbot.waitUntil(lambda: w._active_open is None and not w._open_queue, timeout=30_000)
    qtbot.wait(30)
    doc_id = live_doc_id(w._session, csv)
    assert w._session.calibration(doc_id, 0).note == NOTE

    proj = tmp_path / "p.nanoscope"
    assert w._write_project(str(proj))
    sidecar_path(csv).unlink()                      # the sidecar is gone; only the project knows

    w2 = _window(qtbot)
    with qtbot.waitSignal(w2.projectRestored, timeout=30_000):
        w2._open_project(str(proj))
    qtbot.wait(50)
    live = live_doc_id(w2._session, csv)
    assert w2._session.calibration(live, 0).note == NOTE
    w._controller.abort()
    w2._controller.abort()


def test_reset_to_default_still_ends_the_note(qapp, qtbot, tmp_path):
    """The rule is NOT "always keep the note".

    A note is carried when the writer had nothing to say about it. *Reset to default* has
    something to say: this channel has no calibration, so it has no note either, and the
    sidecar entry goes with it. A fix that resurrected the note here would leave the user
    with "measured against the 1 MΩ standard" attached to gain 1.0.
    """
    csv = tmp_path / "pore.csv"
    arr = _trace()
    _write_csv(csv, arr)
    _write_sidecar(csv, gain=0.5, note=NOTE)

    s = Session()
    doc_id = s.add_document(_doc(csv, arr))
    assert s.calibration(doc_id, 0).note == NOTE
    panel = _panel_for(qtbot, s, doc_id)

    panel._reset()

    assert s.calibration(doc_id, 0).note == ""
    assert not sidecar_path(csv).exists()


# ------------------------------------------------------------------------------------------------
# §2.5, second half — the note that goes while the FILE stays
# ------------------------------------------------------------------------------------------------
def _two_channel_doc(path: Path, arr: np.ndarray) -> TraceDocument:
    """A recording with a second, separately-calibrated channel — which is what keeps the
    sidecar alive when the first one is reset, and therefore what makes the loss silent."""
    cur = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    volt = ChannelInfo(index=1, name="IN14", raw_units="mV", kind=ChannelKind.VOLTAGE)
    meta = RecordingMeta(path=path, format_name="synthetic", n_samples=arr.size,
                         sample_rate_hz=FS, rate_provenance=Provenance.FILE_META,
                         channels=(cur, volt))
    return TraceDocument(meta=meta, channels=(arr, np.full(arr.size, -6000.0,
                                                           dtype=np.float32)))


def _noted_two_channel_session(tmp_path) -> tuple[Session, str, Path]:
    csv = tmp_path / "pore.csv"
    arr = _trace()
    _write_csv(csv, arr)
    s = Session()
    doc_id = s.add_document(_two_channel_doc(csv, arr))
    s.set_calibration(doc_id, 0, ChannelCalibration(gain=0.5, offset=0.0, target_units="pA",
                                                    note=NOTE))
    # 0.2 is NOT this voltage channel's default (0.1), so the sidecar survives ch0's reset.
    s.set_calibration(doc_id, 1, ChannelCalibration(gain=0.2, offset=0.0, target_units="mV"))
    assert _note_on_disk(sidecar_path(csv)) == NOTE
    return s, doc_id, csv


def test_a_reset_that_keeps_the_file_still_asks_before_erasing_the_note(qapp, qtbot, tmp_path):
    """Measured on the tree as found: the sidecar SURVIVED, the note was erased out of it,
    ``sidecar_effect`` said WRITTEN so the removal confirmation never fired, and the status
    line read *"Applied · saved to pore.csv.npcal.json"* — which is what a successful save
    looks like. The file is not the thing that cannot be recreated; the note is.
    """
    s, doc_id, csv = _noted_two_channel_session(tmp_path)
    panel = _panel_for(qtbot, s, doc_id)
    asked: list = []
    panel._confirm_note_loss = lambda path, lost, **kw: (asked.append((path, lost)), True)[1]

    panel._reset()

    assert asked, "the note was erased from a surviving file with nothing asked"
    path, lost = asked[0]
    assert path == sidecar_path(csv)
    assert lost == ((0, NOTE),), lost
    assert sidecar_path(csv).exists(), "the file itself was not the thing at stake"
    assert "1" in json.loads(sidecar_path(csv).read_text())["channels"]


def test_declining_that_prompt_changes_nothing_at_all(qapp, qtbot, tmp_path):
    """A confirmation the user cannot decline is decoration. Both halves: the file keeps the
    note, and the live calibration is not quietly reset behind it."""
    s, doc_id, csv = _noted_two_channel_session(tmp_path)
    panel = _panel_for(qtbot, s, doc_id)
    panel._confirm_note_loss = lambda path, lost, **kw: False

    panel._reset()

    assert _note_on_disk(sidecar_path(csv)) == NOTE
    assert s.calibration(doc_id, 0).gain == 0.5
    assert s.calibration(doc_id, 0).note == NOTE
    assert "Nothing changed" in panel._status.text()


def test_an_ordinary_apply_that_keeps_the_note_asks_nothing(qapp, qtbot, tmp_path):
    """The other failure mode of a new prompt: firing on the ordinary edit. *Apply* carries
    the note forward, so it destroys nothing and must stay silent — a confirmation the user
    sees on every gain change is one they learn to click through, and the same OK button
    then answers the one that matters."""
    s, doc_id, csv = _noted_two_channel_session(tmp_path)
    panel = _panel_for(qtbot, s, doc_id)
    asked: list = []
    panel._confirm_note_loss = lambda path, lost, **kw: (asked.append(lost), True)[1]

    panel._gain.setText("0.75")
    panel._apply()

    assert asked == []
    assert _note_on_disk(sidecar_path(csv)) == NOTE
    assert s.calibration(doc_id, 0).gain == 0.75


def test_the_deletion_prompt_names_the_button_that_raised_it(qapp, qtbot, tmp_path,
                                                             monkeypatch):
    """The one prompt guarding the sidecar was titled *"Reset calibration"* unconditionally —
    including when *Apply* raised it, which it does whenever the typed values happen to equal
    the defaults. A destructive prompt that names a button the user did not press reads as
    the app talking about something else, and the natural response is to accept it unread.

    Asserted on what is ON SCREEN. macOS discards a ``QMessageBox`` window title outright, so
    a fix that only corrected the title would be correct and invisible on the one platform
    this app runs on — the tooltip-on-a-native-menu-item lesson, again.
    """
    csv = tmp_path / "pore.csv"
    arr = _trace()
    _write_csv(csv, arr)
    _write_sidecar(csv, gain=0.5, note=NOTE)
    s = Session()
    doc_id = s.add_document(_doc(csv, arr))

    panel = CalibrationPanel(s)          # the REAL confirmation, not the injected seam
    qtbot.addWidget(panel)
    panel.set_selection(doc_id, 0)
    seen: list[list[str]] = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec",
                        lambda self: seen.append([self.text(), self.informativeText()]
                                                 + [b.text() for b in self.buttons()]))

    panel._gain.setText("1.0")           # back at the default -> Apply deletes the sidecar
    panel._offset.setText("0.0")
    panel._units.setText("")
    panel._apply()

    assert seen, "Apply deleted the sidecar without asking"
    visible = seen[0]
    assert any("Apply" in s for s in visible), visible
    assert not any("Reset" in s for s in visible), visible
    assert "note" in visible[1] and str(sidecar_path(csv)) in visible[1]
    # exec() returned without a click, so nothing was confirmed and nothing was deleted
    assert sidecar_path(csv).exists()


def test_a_failed_sidecar_write_reports_the_reason_the_OS_gave(qapp, qtbot, tmp_path,
                                                               monkeypatch):
    """It said "(the folder is not writable)" for every ``OSError`` — one of at least five
    things that error means here, and the one the user cannot act on when the folder is in
    fact writable. A guess in a failure message costs more than no explanation."""
    csv = tmp_path / "pore.csv"
    arr = _trace()
    _write_csv(csv, arr)
    s = Session()
    doc_id = s.add_document(_doc(csv, arr))
    panel = _panel_for(qtbot, s, doc_id)

    from nanoscope.core import atomic_write
    monkeypatch.setattr(atomic_write.os, "replace",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            OSError(28, "No space left on device")))
    panel._gain.setText("0.5")
    panel._apply()

    status = panel._status.text()
    assert "space" in status.lower(), status
    assert "not writable" not in status.lower(), status
    assert "NOT stored next to your data" in status, status


# ================================================================================================
# §2.8 — a gain that destroys the channel, and an Issues box that knows about it
# ================================================================================================
def test_a_zero_gain_is_refused_at_the_value_object(qapp, tmp_path):
    """Measured through ``Session.calibrated`` on the owner's real ABF: ``gain=0`` gave
    ``non-finite 0/2993798``, ``meta.issues=()`` and ``band_rms = 0.0000`` — C4's exact
    original symptom, because 0.0 is finite and the P0 non-finite gate never sees it."""
    with pytest.raises(ValueError) as exc:
        ChannelCalibration(gain=0.0)
    assert "zero" in str(exc.value) and "offset" in str(exc.value)
    # underflow to zero in float32 is the same destruction by another route
    with pytest.raises(ValueError):
        ChannelCalibration(gain=1e-60)


def test_a_negative_gain_is_still_allowed(qapp):
    """Headstages invert. A sign flip loses nothing and must not be caught by the zero rule."""
    assert ChannelCalibration(gain=-1.0).gain == -1.0
    assert ChannelCalibration(gain=-0.5).apply(
        np.array([2.0, -4.0], dtype=np.float32)).tolist() == [-1.0, 2.0]


def test_a_hand_edited_sidecar_with_a_zero_gain_is_rejected_with_an_issue(qapp, tmp_path):
    """The four ways in, at the one door. A sidecar is a hand-editable file."""
    csv = tmp_path / "pore.csv"
    arr = _trace()
    _write_csv(csv, arr)
    sidecar_path(csv).write_text(json.dumps({"version": 1, "channels": {
        "0": {"gain": 0.0, "offset": 0.0, "target_units": "pA", "note": ""}}}))

    s = Session()
    doc_id = s.add_document(_doc(csv, arr))
    assert s.calibration(doc_id, 0).gain == 1.0, "a zero gain was installed"
    assert any("malformed" in i for i in s.document(doc_id).meta.issues), \
        s.document(doc_id).meta.issues


def test_the_panel_says_WHY_a_zero_gain_cannot_be_committed(qapp, qtbot, tmp_path):
    """"Locked" is styling, not ``setEnabled(False)`` — and a refusal is not a blank.

    The preview said "gain/offset: enter a number" for a perfectly well-formed ``0``, which
    is not what is wrong with it."""
    csv = tmp_path / "pore.csv"
    arr = _trace()
    _write_csv(csv, arr)
    s = Session()
    doc_id = s.add_document(_doc(csv, arr))
    panel = _panel_for(qtbot, s, doc_id)

    panel._gain.setText("0")
    assert panel._pending() is None
    text = panel._formula.text()
    assert "gain 0" in text and "deletes the recording" in text, text
    panel._apply()
    assert s.calibration(doc_id, 0).gain == 1.0


def test_a_calibration_that_overflows_float32_reaches_the_Issues_box(qapp, qtbot, tmp_path):
    """The second hole. Non-finite values created AFTER load never reached ``meta.issues``,
    while the PSD panel's text told the user to go read the Issues box — which said "(none)".
    Measured on the real ABF: gain 1.5e36 → 2,420 non-finite calibrated samples,
    ``meta.issues = ()``."""
    csv = tmp_path / "pore.csv"
    arr = _trace()
    _write_csv(csv, arr)
    s = Session()
    doc_id = s.add_document(_doc(csv, arr))

    info_panel = FileInfoPanel(s)
    qtbot.addWidget(info_panel)
    info_panel.set_document(doc_id)
    assert info_panel._issues.toPlainText() == "(none)"

    s.set_calibration(doc_id, 0, ChannelCalibration(gain=1e37), write_sidecar=False)
    chan = s.calibrated(doc_id, 0)

    assert not np.isfinite(np.asarray(chan.data)).all()
    assert chan.issues, "the calibrated channel does not know it was ruined"
    assert "AFTER calibration" in chan.issues[0]
    assert info_panel._issues.toPlainText() != "(none)", \
        "the box two panels point at is still empty"
    assert "AFTER calibration" in info_panel._issues.toPlainText()


def test_a_healthy_calibration_adds_nothing_to_the_issues_box(qapp, qtbot, tmp_path):
    """Silence has to mean "nothing wrong", so an ordinary gain must stay silent — and must
    not pay for a scan of the calibrated array to find that out."""
    csv = tmp_path / "pore.csv"
    arr = _trace()
    _write_csv(csv, arr)
    s = Session()
    doc_id = s.add_document(_doc(csv, arr))
    s.set_calibration(doc_id, 0, ChannelCalibration(gain=0.25, offset=3.0), write_sidecar=False)
    assert s.calibrated(doc_id, 0).issues == ()
    assert s.calibration_issues(doc_id, 0) == ()


def test_a_channel_the_LOAD_already_reported_is_not_reported_twice(qapp, tmp_path):
    """``meta.issues`` and ``CalibratedChannel.issues`` add up; they must not double-count."""
    arr = _trace()
    arr[17] = np.nan
    csv = tmp_path / "pore.csv"
    _write_csv(csv, arr)
    s = Session()
    doc_id = s.add_document(_doc(csv, arr))
    assert any("non-finite" in i for i in s.document(doc_id).meta.issues)
    s.set_calibration(doc_id, 0, ChannelCalibration(gain=2.0), write_sidecar=False)
    assert s.calibrated(doc_id, 0).issues == ()


def test_a_constant_channel_is_named_rather_than_reported_as_a_perfect_pore(qapp, qtbot,
                                                                            tmp_path):
    """``band_rms`` returns a literal, correct ``0.0`` for a signal with no variance — and on
    screen that is indistinguishable from a flawless pore. ``band_rms`` cannot tell the two
    apart; the panel still has the samples, so it can."""
    from nanoscope.gui.psd_panel import PSDWindow

    # Live for its first half, then clamped flat — a stuck ADC or a headstage that lost
    # contact part way through. A wholly flat channel is classified DEAD at load and never
    # reaches the PSD list at all, so the reachable case is a flat WINDOW.
    half = N // 2
    arr = _trace()
    arr[half:] = -34.0
    csv = tmp_path / "flat.csv"
    _write_csv(csv, arr)
    s = Session()
    doc_id = s.add_document(_doc(csv, arr))

    win = PSDWindow(s)
    qtbot.addWidget(win)
    win.set_target(doc_id, 0)
    qtbot.wait(30)
    win._start_spin.setValue(half / FS)
    win._end_spin.setValue(N / FS)
    win._recompute()
    qtbot.wait(30)

    curve = win._compute_curve(win._recordings[0], "#000000")
    assert curve.constant is True
    assert curve.summary["total_rms"] == 0.0
    # Every PSD bin of a signal with no variance is 0, which log-log cannot draw — and the
    # panel used to report that as "No positive spectral points to plot (check the log-log
    # toggle)", sending the user to fiddle with an axis setting while their channel was a
    # flat line. Same misdirection the NaN branch already corrects.
    shown = win._message_item.toPlainText()
    assert "log-log toggle" not in shown, shown
    assert "same value" in shown and "not because the pore is quiet" in shown, shown

    # With the log axes OFF the all-zero spectrum IS plottable, so the curve is drawn and
    # the Details pane prints "Total RMS noise: 0 pA" — the reachable form of C4's original
    # symptom, and the one a screenshot in a lab notebook would carry.
    win._loglog_check.setChecked(False)
    win._recompute()
    qtbot.wait(30)
    details = win._curve_details(win._curves[0], header=False)
    assert "Total RMS noise:   0 pA" in details, details
    assert "arithmetic, not a noise measurement" in details, details


# ================================================================================================
# §2.10 — a partial figure export
# ================================================================================================
def _seed_previous_export(stem: Path, names) -> None:
    for name in names:
        for ext in ("pdf", "png", "csv"):
            stem.with_name(f"{stem.name}_{name}.{ext}").write_bytes(b"OLD")


def test_a_figure_the_export_could_not_build_is_named_in_a_dialog(qapp, qtbot, tmp_path,
                                                                  monkeypatch):
    """Consent said twelve files would be replaced. Nine failing left three from today beside
    nine from a previous export, under one ``figures_*`` stem, with no marker — disclosed
    only by a status line that clears after twelve seconds."""
    w = _window(qtbot)
    stem = tmp_path / "figures"
    _seed_previous_export(stem, ["dwell", "depth"])

    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec",
                        lambda self: shown.append((self.text(), self.informativeText())))

    w._report_unreplaced_figures(stem, ["depth"])

    assert shown, "a stale file under this export's stem was disclosed by nothing"
    text, informative = shown[0]
    assert "earlier export" in text
    for ext in ("pdf", "png", "csv"):
        assert f"figures_depth.{ext}" in informative, informative
    assert "figures_dwell" not in informative, "a file that WAS replaced was named as stale"


def test_a_figure_that_failed_with_no_file_of_that_name_needs_no_dialog(qapp, qtbot, tmp_path,
                                                                        monkeypatch):
    """The two situations are different and the user needs them told apart: "not exported"
    is a status line, "left over from an earlier export under this name" is a dialog."""
    w = _window(qtbot)
    shown: list = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: shown.append(self.text()))

    w._report_unreplaced_figures(tmp_path / "figures", ["depth"])

    assert shown == []
    assert "could not be built" in w.statusBar().currentMessage()


# ================================================================================================
# §2.7 — through the save the user actually presses
# ================================================================================================
def test_saving_a_project_over_a_symlink_tells_the_user_which_file_changed(qapp, qtbot,
                                                                           tmp_path,
                                                                           monkeypatch):
    csv = tmp_path / "pore.csv"
    arr = _trace()
    _write_csv(csv, arr)
    w = _window(qtbot)
    w._session.add_document(_doc(csv, arr))

    real = tmp_path / "real.nanoscope"
    save_project(real, ProjectState(documents=[DocumentRef(path=str(csv.resolve()))]))
    link = tmp_path / "link.nanoscope"
    link.symlink_to(real)

    shown: list[str] = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **kw: shown.append(a[2])))
    # This test deliberately Save-As overwrites an existing project. The native chooser
    # would have asked first in production; the inter-process guard asks again for the
    # exact revision so a writer landing after that chooser cannot be lost.
    monkeypatch.setattr(w, "_confirm_overwrite_existing_project", lambda path: True)

    assert w._write_project(str(link))

    assert link.is_symlink()
    assert shown and "link.nanoscope" in shown[0] and str(real) in shown[0], shown


def test_the_symlink_note_does_not_interrupt_every_single_save(qapp, qtbot, tmp_path,
                                                               monkeypatch):
    """A standing arrangement is news once, not on every Ctrl+S.

    The note is true on every save through that link, so the naive version puts a modal in
    front of the save shortcut for the rest of the session — which is precisely how a user
    learns to dismiss this app's dialogs unread, and the sidecar-deletion confirmation is
    one of them. Said once per file; a note that was never said still interrupts.
    """
    csv = tmp_path / "pore.csv"
    arr = _trace()
    _write_csv(csv, arr)
    w = _window(qtbot)
    w._session.add_document(_doc(csv, arr))

    real = tmp_path / "real.nanoscope"
    save_project(real, ProjectState(documents=[DocumentRef(path=str(csv.resolve()))]))
    link = tmp_path / "link.nanoscope"
    link.symlink_to(real)

    shown: list[str] = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **kw: shown.append(a[2])))
    monkeypatch.setattr(w, "_confirm_overwrite_existing_project", lambda path: True)

    assert w._write_project(str(link))
    assert len(shown) == 1
    assert w._write_project(str(link))
    assert w._write_project(str(link))
    assert len(shown) == 1, f"the same sentence interrupted {len(shown)} saves"
    assert link.is_symlink()

    # A DIFFERENT link is a different arrangement and is disclosed on its own.
    link2 = tmp_path / "other.nanoscope"
    link2.symlink_to(real)
    assert w._write_project(str(link2))
    assert len(shown) == 2 and "other.nanoscope" in shown[1], shown
