"""S8-DATA2 — an export does not write a table the app has itself flagged stale.

MEASURED DEFECT (sweep 8, a 2 s / 50 kHz slice of the owner's -600 mV recording as a
'time,current' CSV). After Inspector > File Info > Sample rate / time unit... the event table
on screen wears the amber line ("Times stale -- sample rate changed") and Analysis > Export
events went ahead anyway: no dialog, status "Wrote 10 accepted event(s) of 10 ...". The CSV is,
correctly for a stale table, still on the OLD clock (``t_start_s`` and ``dwell_s`` are
``start_idx / fs`` at the rate the detector ran at); the ``.meta.json`` beside it was built from
two places -- ``fs`` from the table's own provenance (the OLD rate) but ``t0_s`` and the
``clock`` note from the LIVE recording (``scoring.clock_stamp(meta=...)``, the DECLARED one).
The note tells the reader to "add t0 to a number here to find it on the trace", and for the
declared 40 kHz over a 50 kHz detection the first event landed 64.98 ms (2599 samples) off. The
unit declaration did the same to a bare-millisecond CSV read at 50 Hz: the same CSV exported
before and after the declaration differed only in ``t0_s`` (10000.0 -> 10.0) while ``fs`` and
every time in it were still the 50 Hz reading. Neither file said the table was stale, that a
declaration existed, or what the live rate was. The other two scoring exports that stamp the
clock (``Export events for training`` and ``Export sub-state levels``) build their sidecar the
same two-source way and took the same stale table without a word.

THE FIX. ``MainWindow._refuse_stale_export`` asks the question the amber line already answers
(``AnalysisPanel._sample_rate_staleness``: the ``fs`` the detector stamped in the table's
provenance against the recording's live rate) at the top of the three doors, BEFORE any file
dialog or selection box, and refuses with one status-bar sentence that names both rates and the
way forward (re-run detection under the declared rate). A status line and not a modal, so a
scripted or batch export is never left waiting on a button; nothing is written, so no CSV
exists beside a sidecar that describes another clock. The table is not repaired or re-scaled:
the one repair that is true is a re-run, and it is one click on the banner. An export of a table
that is NOT stale takes exactly the path it took before -- the sidecar is unchanged byte for
byte, and ``fs`` and the clock block are, there, the same time base by construction (they agree
to 1e-9), which the "way forward" tests pin from the reader's side: ``t0_s + t_start_s`` lands
on the trace sample the event starts at. A recording that has NO rate any more (the time base
handed back to a file that declares none) has no clock to splice, and no declared rate to re-run
under: that export is left as it was, with the origin said to be unknown (``t0_s_known: false``).

These tests drive the real ``MainWindow``: a real Detect on a synthetic recording, the real
declaration (``Session.set_sample_rate_declaration``, the call the dialog makes), the real
``Analysis`` menu actions triggered the way a click does, the banner's own Re-run button, and
the files the export left on disk. The file dialog and every ``QMessageBox`` are recorded so a
refusal that opened either fails loudly.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from nanoscope.analysis import scoring
from nanoscope.analysis.events import EventSet
from nanoscope.core.document import TraceDocument
from nanoscope.core.sample_rate import SampleRateDeclaration
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance, RecordingMeta
from nanoscope.gui import main_window as mw
from nanoscope.gui.main_window import MainWindow

FS = 100_000.0
N = 100_000
#: A recording that does not start at zero, so an origin taken from the wrong place shows.
T0 = 0.05
_EVENTS = [(10_000, 10_300), (30_000, 30_250), (50_000, 50_400),
           (70_000, 70_200), (85_000, 85_350)]

#: door -> the Analysis menu action a click on it triggers.
DOORS = {
    "events": "_act_export_events",
    "training": "_act_export_training",
    "levels": "_act_export_levels",
}


# ================================================================================================
# builders
# ================================================================================================
def _trace(seed: int = 0) -> np.ndarray:
    """A quiet 100 pA baseline with five unmistakable 30 pA blockades -- enough for Detect, and
    for CUSUM to find a level in each."""
    rng = np.random.default_rng(seed)
    x = np.full(N, 100.0) + rng.normal(0.0, 1.5, N)
    for s, e in _EVENTS:
        x[s:e] -= 30.0
    return np.ascontiguousarray(x, dtype=np.float32)


def _doc(path: Path, *, rate: float = FS, prov: Provenance = Provenance.FILE_META,
         fmt: str = "synthetic") -> TraceDocument:
    arr = _trace()
    info = ChannelInfo(index=0, name="current", raw_units="pA", kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=path, format_name=fmt, n_samples=arr.size, sample_rate_hz=rate,
                         rate_provenance=prov, channels=(info,), t0_s=T0)
    return TraceDocument(meta=meta, channels=(arr,))


def _bare_ms_doc(path: Path) -> TraceDocument:
    """A 100 kHz trace whose time column was in MILLISECONDS under a bare header: it reads as
    100 Hz with its start 0.05 s, until the user declares the unit."""
    return _doc(path, rate=100.0, prov=Provenance.MEASURED, fmt="Text trace")


@pytest.fixture
def window(qapp, qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()
    w._dirty = False


@pytest.fixture
def rig(window, monkeypatch, tmp_path):
    """What an export door would have raised: every file dialog asked, every modal shown."""
    seen = SimpleNamespace(asked=[], modals=[], dest=tmp_path / "out.csv")

    def ask(*args, **kwargs):
        seen.asked.append(kwargs.get("title", ""))
        return seen.dest

    def exec_(self):
        seen.modals.append(self.text())
        return 0

    def static_modal(name):
        def raised(*args, **kwargs):
            seen.modals.append(f"{name}: {args[2] if len(args) > 2 else ''}")
            return QtWidgets.QMessageBox.Ok
        return staticmethod(raised)

    monkeypatch.setattr(mw, "ask_export_path", ask)
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", exec_)
    # The training door's own selection box is answered with its middle choice.
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "clickedButton",
        lambda self: next((b for b in self.buttons() if "Accepted candidates" in b.text()), None))
    for name in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QtWidgets.QMessageBox, name, static_modal(name))
    return seen


def _detect(window, qtbot) -> None:
    panel = window._analysis_panel
    panel.detection_panel.min_dwell_s.setValue(5e-4)
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        panel.trigger_detection()
    qtbot.wait(20)       # the queued add_result -> resultsChanged -> table settles


def _fit_levels(window, qtbot) -> None:
    window._fit_levels()
    qtbot.waitUntil(lambda: not window._analysis_controller.is_busy(), timeout=10_000)
    window._sync_export_action()
    assert window._level_fit is not None and not window._level_fit.payload.empty, (
        "the synthetic recording must give CUSUM levels to export")


def _open(window, qtbot, doc: TraceDocument, *, door: str = "events") -> str:
    """Add the recording the way the load pipeline does, select it, Detect on it (and, for the
    levels door, fit levels on the table) -- the state a user is in before declaring."""
    key = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(key, 0)
    _detect(window, qtbot)
    res = window._analysis_panel.current_detection()
    assert res is not None and len(res.payload.df) >= 3
    if door == "levels":
        _fit_levels(window, qtbot)
    return key


def _declare_rate(window, key: str, hz: float) -> None:
    window._session.set_sample_rate_declaration(
        key, SampleRateDeclaration("sample_rate", sample_rate_hz=hz))


def _declare_unit(window, key: str, unit: str) -> None:
    window._session.set_sample_rate_declaration(
        key, SampleRateDeclaration("time_unit", time_unit=unit))


def _fire(window, door: str, rig, dest: Path | None = None) -> None:
    """Click the Analysis menu item for ``door`` (a disabled action would swallow the click and
    make every assertion after it vacuous, so that is asserted first)."""
    action = getattr(window, DOORS[door])
    assert action.isEnabled(), f"{action.text()!r} is disabled -- the click would do nothing"
    rig.asked.clear()
    rig.modals.clear()
    rig.dest = dest or rig.dest
    action.trigger()


def _sidecar(dest: Path) -> dict:
    return json.loads(dest.with_suffix(".meta.json").read_text())


def _assert_refused(window, rig, dest: Path, *, was: str, now: str) -> None:
    said = window.statusBar().currentMessage()
    assert not dest.exists() and not dest.with_suffix(".meta.json").exists(), (
        "a refused export writes nothing -- not the CSV, not a sidecar")
    assert rig.asked == [], "no save dialog for a file that cannot be written"
    assert rig.modals == [], f"a modal would hang a scripted export: {rig.modals}"
    assert said.startswith("Nothing was exported"), said
    assert was in said and now in said, f"both rates are named: {said!r}"
    assert "re-run detection under the declared rate" in said, f"the way forward: {said!r}"
    assert said.index("re-run detection") < said.index(was), "the way forward comes first"


# ================================================================================================
# the refusal -- one test per way the clock can move, then the two sibling doors
# ================================================================================================
def test_export_events_refuses_a_table_a_declared_rate_made_stale(window, qtbot, rig, tmp_path):
    """The finding's case A: detected at 100 kHz, then the recording is declared 50 kHz. The
    amber line says every time in the table is on the old rate; the export used to write that
    table with t0_s and the clock note of the declared recording."""
    key = _open(window, qtbot, _doc(tmp_path / "pore.csv"))
    panel = window._analysis_panel
    _declare_rate(window, key, 50_000.0)
    assert panel._stale.kind == "sample_rate" and not panel._stale.isHidden(), (
        "precondition: the app itself flags this table")

    _fire(window, "events", rig)

    _assert_refused(window, rig, rig.dest, was="100,000 Hz", now="50,000 Hz")


def test_export_events_refuses_a_table_a_declared_time_unit_made_stale(window, qtbot, rig,
                                                                        tmp_path):
    """The finding's case B: a bare-millisecond column read as 100 Hz and detected at that
    rate; declaring the unit moves the rate to 100 kHz AND the start time (0.05 s -> 0.05 ms).
    Before the fix the same CSV came out with a different t0_s and the old fs -- two exports of
    one table that disagree about where the recording starts."""
    key = _open(window, qtbot, _bare_ms_doc(tmp_path / "bare_ms.csv"))
    first = tmp_path / "as_read.csv"
    _fire(window, "events", rig, first)
    assert first.exists(), "precondition: the table exports at the rate it was detected at"
    as_read = _sidecar(first)
    assert as_read["fs"] == pytest.approx(100.0) and as_read["t0_s"] == pytest.approx(T0)
    before = (first.read_bytes(), first.with_suffix(".meta.json").read_bytes())

    _declare_unit(window, key, "ms")
    assert window._session.document(key).meta.sample_rate_hz == pytest.approx(100_000.0)
    assert window._analysis_panel._stale.kind == "sample_rate"
    second = tmp_path / "declared.csv"
    _fire(window, "events", rig, second)

    _assert_refused(window, rig, second, was="100 Hz", now="100,000 Hz")
    assert (first.read_bytes(), first.with_suffix(".meta.json").read_bytes()) == before, (
        "the earlier export is left exactly as it was")


def test_export_events_for_training_refuses_a_stale_table_before_asking_anything(
        window, qtbot, rig, tmp_path):
    """``Export events for training`` shares the two-source sidecar (provenance ``fs`` beside
    the live recording's origin). It opens a selection box and then maybe two warnings before
    it asks for a path: the refusal comes before ALL of them."""
    key = _open(window, qtbot, _doc(tmp_path / "pore.csv"))
    _declare_rate(window, key, 50_000.0)

    _fire(window, "training", rig)

    _assert_refused(window, rig, rig.dest, was="100,000 Hz", now="50,000 Hz")


def test_export_sub_state_levels_refuses_a_stale_table(window, qtbot, rig, tmp_path):
    """``Export sub-state levels`` builds ``fs`` from the detection on screen and the origin
    from the live recording. The level table survives a declaration (nothing drops it), so the
    menu item is live over a table whose times are on the old clock."""
    key = _open(window, qtbot, _doc(tmp_path / "pore.csv"), door="levels")
    _declare_rate(window, key, 50_000.0)
    assert window._level_fit is not None, "precondition: the declaration leaves the level fit"

    _fire(window, "levels", rig)

    _assert_refused(window, rig, rig.dest, was="100,000 Hz", now="50,000 Hz")


def test_the_refusal_is_the_ambers_judgement_not_a_flag_that_a_declaration_sets(
        window, qtbot, rig, tmp_path):
    """Declaring a rate that is the rate the table was detected at changes no time in it: the
    amber line stays down and the export is not refused. (The first version of any guard keyed
    on "a declaration exists" would lock out exactly the user who confirmed the file's rate.)"""
    key = _open(window, qtbot, _doc(tmp_path / "pore.csv"))
    _declare_rate(window, key, FS)
    assert window._analysis_panel._stale.isHidden(), "precondition: nothing moved"

    _fire(window, "events", rig)

    assert rig.dest.exists() and rig.dest.with_suffix(".meta.json").exists()
    assert window.statusBar().currentMessage().startswith("Wrote "), (
        window.statusBar().currentMessage())


# ================================================================================================
# the way forward is real -- and it is exactly the half of the sidecar the defect broke
# ================================================================================================
@pytest.mark.parametrize("door", ["events", "training", "levels"])
def test_re_running_under_the_declared_rate_lifts_the_refusal_and_the_sidecar_agrees(
        door, window, qtbot, rig, tmp_path):
    """The sentence's own advice, followed: the banner's Re-run, then the same menu item. The
    export is written, ``fs`` is the declared rate, and the origin the sidecar stamps is the
    one the reader needs -- ``t0_s + t_start_s`` is the instant the trace axis shows at the
    event's first sample, which is the promise the clock note makes and the stale export broke."""
    key = _open(window, qtbot, _doc(tmp_path / "pore.csv"), door=door)
    panel = window._analysis_panel
    _declare_rate(window, key, 50_000.0)
    _fire(window, door, rig)
    assert not rig.dest.exists(), "precondition: it was refused"

    with qtbot.waitSignal(window._analysis_controller.finished, timeout=60_000):
        panel._stale._btn.click()                     # the banner's own Re-run
    qtbot.wait(20)
    assert panel._stale.isHidden(), "precondition: the table is current again"
    if door == "levels":
        _fit_levels(window, qtbot)      # a re-detection retires the level fit with its table
    _fire(window, door, rig)

    assert rig.dest.exists(), window.statusBar().currentMessage()
    meta = _sidecar(rig.dest)
    live = window._session.document(key).meta
    assert meta["fs"] == pytest.approx(50_000.0) == pytest.approx(live.sample_rate_hz)
    assert meta[scoring.CLOCK_KEY] == pytest.approx(T0)
    assert meta[scoring.CLOCK_BLOCK_KEY]["t0_s_known"] is True
    if door == "levels":
        return                          # level rows carry no sample index to place
    table = pd.read_csv(rig.dest)
    on_the_trace = window._session.x_axis(key).values[table["start_idx"].to_numpy(dtype=int)]
    np.testing.assert_allclose(meta[scoring.CLOCK_KEY] + table["t_start_s"].to_numpy(),
                               on_the_trace, atol=1e-9)


def test_handing_the_time_base_back_lifts_the_refusal_and_restores_the_same_export(
        window, qtbot, rig, tmp_path):
    """The refusal follows the live judgement, so it is not sticky: 'Use the file's own time
    base' makes the table current again, and the export is then byte-for-byte the one made
    before the declaration."""
    key = _open(window, qtbot, _bare_ms_doc(tmp_path / "bare_ms.csv"))
    before = tmp_path / "before.csv"
    _fire(window, "events", rig, before)
    _declare_unit(window, key, "ms")
    _fire(window, "events", rig, tmp_path / "refused.csv")
    assert not (tmp_path / "refused.csv").exists(), "precondition: it was refused"

    window._session.set_sample_rate_declaration(key, None)
    assert window._analysis_panel._stale.isHidden()
    after = tmp_path / "after.csv"
    _fire(window, "events", rig, after)

    assert after.read_bytes() == before.read_bytes()
    assert (after.with_suffix(".meta.json").read_bytes()
            == before.with_suffix(".meta.json").read_bytes())


# ================================================================================================
# what the guard must not touch
# ================================================================================================
def test_a_table_that_is_not_stale_is_written_exactly_as_the_writer_writes_it(
        window, qtbot, rig, tmp_path):
    """The door adds nothing to a current table's export: its two files are byte-identical to
    what ``write_events_export`` produces for the same rows, provenance and recording."""
    _open(window, qtbot, _doc(tmp_path / "pore.csv"))
    res = window._analysis_panel.current_detection()
    accepted = res.payload.accepted.df
    source_file, doc_meta = window._export_source(res.doc_id)
    direct = scoring.write_events_export(
        accepted, tmp_path / "direct.csv",
        provenance=window._export_provenance(res.payload.provenance, accepted),
        source_file=source_file, detector=res.module_id, meta=doc_meta,
        n_total=len(res.payload.df))

    _fire(window, "events", rig, tmp_path / "door.csv")

    assert (tmp_path / "door.csv").read_bytes() == direct["data"].read_bytes()
    assert (tmp_path / "door.meta.json").read_bytes() == direct["meta"].read_bytes()
    assert window.statusBar().currentMessage().startswith(
        f"Wrote {len(accepted)} accepted event(s) of {len(res.payload.df)} to door.csv")


def test_a_table_with_no_rate_stamp_is_still_exported(window, qtbot, rig, tmp_path):
    """A table from before the detectors stamped ``fs`` has nothing to be judged against (the
    amber line makes no RATE claim about it; it may speak of the estimator instead, because the
    estimator's fingerprint moves with the rate), so the door does not lock it out."""
    key = _open(window, qtbot, _doc(tmp_path / "pore.csv"))
    res = window._analysis_panel.current_detection()
    prov = dict(res.payload.provenance)
    prov.pop("fs")
    window._session.add_result(replace(res, payload=EventSet(res.payload.df, prov)))
    _declare_rate(window, key, 50_000.0)
    stale = window._analysis_panel._stale
    assert stale.kind != "sample_rate" or stale.isHidden(), (
        f"precondition: no stamp, no rate judgement -- {stale.kind}: {stale.message()}")

    _fire(window, "events", rig)

    assert rig.dest.exists()
    assert _sidecar(rig.dest)["fs"] is None


def test_a_table_whose_recording_lost_its_rate_still_exports_with_an_honest_unknown_origin(
        window, qtbot, rig, tmp_path):
    """The amber line calls this table stale too (100,000 Hz -> unknown), but there is no live
    clock to splice onto it: the sidecar names the table's own rate and says the origin is not
    known. Refusing would tell the user to "re-run under the declared rate" when the
    declaration is the very thing they just took away -- and Detect is gated without a rate."""
    doc = _doc(tmp_path / "rateless.csv", rate=None, prov=Provenance.UNKNOWN, fmt="Text trace")
    key = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(key, 0)
    _declare_rate(window, key, FS)
    _detect(window, qtbot)
    assert window._analysis_panel.current_detection() is not None
    window._session.set_sample_rate_declaration(key, None)      # "Use the file's own time base"
    assert window._session.document(key).meta.sample_rate_hz is None
    assert window._analysis_panel._stale.kind == "sample_rate", "precondition: the amber line is up"

    _fire(window, "events", rig)

    assert rig.dest.exists(), window.statusBar().currentMessage()
    meta = _sidecar(rig.dest)
    assert meta["fs"] == pytest.approx(FS)
    assert meta[scoring.CLOCK_KEY] is None
    assert meta[scoring.CLOCK_BLOCK_KEY]["t0_s_known"] is False
