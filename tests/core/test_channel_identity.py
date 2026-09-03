"""C-6 — a channel INDEX is a position, not an identity, and everything persisted about a
channel was filed under the position alone.

Measured on the tree as found: re-export a recording with its signal columns in the other
order — same sample count, same channel count, byte-identical file size, only the sampled
digest differs — and

* ``gain=0.5, offset=-1.0, target_units='pA'`` lands on the **voltage** channel, so the
  -600 mV trace reads **-301.0 pA**;
* the real current channel falls back to gain 1.0 / offset 0.0 / units ``''``;
* an 11-event detection is reattached to the voltage trace;
* and because the result's stored ``calibration_fingerprint`` MATCHES the misplaced live
  calibration, the staleness system positively CERTIFIES those events as current-valid.

On the project path a loud banner fires about the file — it just never mentions channels.
On the ``.npcal.json`` sidecar path there is no project, no fingerprint, no banner and no
``meta.issues``: zero warning, wrong number on screen. That is why the sidecar is the half
that had to be fixed at the same seam.

THIS FILE IS THE PURE HALF: the resolver, the sidecar, and the v7 records. The GUI restore
is pinned in ``tests/gui/test_restore_carry_and_identity.py``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from nanoscope.core import calibration as C
from nanoscope.core.calibration import (
    CalibrationSet,
    ChannelCalibration,
    load_sidecar,
    save_sidecar,
    sidecar_path,
)
from nanoscope.core.types import ChannelInfo, ChannelKind, Provenance


def _api(name: str):
    """Late lookup with a spoken assertion — against a pre-fix package this file must fail
    on its own claims, not on an ImportError at collection."""
    obj = getattr(C, name, None)
    assert obj is not None, f"nanoscope.core.calibration has no {name!r}"
    return obj


def _records_carry_identity() -> None:
    """Assert the four records have the identity fields BEFORE building one with them.

    Constructing ``CalibrationRec(channel_name=...)`` against a package that lacks the
    field raises TypeError, and a TypeError reads like a broken test rather than like the
    defect. Every test below that builds an identity-carrying record calls this first, so
    the failure is a sentence."""
    import dataclasses

    from nanoscope.core.project import CalibrationRec, FitRec, LayerRec, ResultRec

    for cls in (CalibrationRec, LayerRec, ResultRec, FitRec):
        names = {f.name for f in dataclasses.fields(cls)}
        assert {"channel_name", "channel_kind"} <= names, (
            f"{cls.__name__} still keys on the channel index alone")


def _ch(index: int, name: str, kind: ChannelKind) -> ChannelInfo:
    return ChannelInfo(index=index, name=name, raw_units="pA", kind=kind)


VOLT_FIRST = (_ch(0, "voltage", ChannelKind.VOLTAGE), _ch(1, "current", ChannelKind.CURRENT))
CURR_FIRST = (_ch(0, "current", ChannelKind.CURRENT), _ch(1, "voltage", ChannelKind.VOLTAGE))


# ================================================================================================
# the resolver
# ================================================================================================

def test_an_unchanged_recording_resolves_by_index_and_says_nothing():
    resolve = _api("resolve_channel")
    assert resolve(VOLT_FIRST, 1, "current", "current") == (1, None)
    assert resolve(VOLT_FIRST, 0, "voltage", "voltage") == (0, None)


def test_a_record_with_no_identity_behaves_exactly_as_before():
    """Frozen contract 8 in miniature. A pre-v7 project and a version-1 sidecar carry no
    identity, and a warning on every old file is how a real one gets ignored."""
    resolve = _api("resolve_channel")
    assert resolve(VOLT_FIRST, 1) == (1, None)
    assert resolve(VOLT_FIRST, 0) == (0, None)
    assert resolve(VOLT_FIRST, 7) == (None, None)      # out of range: skipped, silently


def test_a_reordered_recording_relocates_by_identity_and_says_so():
    resolve = _api("resolve_channel")
    idx, moved = resolve(CURR_FIRST, 1, "current", "current")
    assert idx == 0, "the record's own channel is still here, one index along"
    assert moved and "current" in moved and "channel 0" in moved and "channel 1" in moved


def test_a_channel_that_is_gone_is_refused_rather_than_applied_by_position():
    """THE DEFECT. The index is still in range and still names a real channel — a different
    one. Installing it there is what read -301.0 pA off the millivolt trace."""
    resolve = _api("resolve_channel")
    only_voltage = (_ch(0, "voltage", ChannelKind.VOLTAGE),)
    idx, why = resolve(only_voltage, 0, "current", "current")
    assert idx is None
    assert why and "current" in why and "voltage" in why


def test_two_channels_with_one_identity_are_refused_not_guessed():
    resolve = _api("resolve_channel")
    twins = (_ch(0, "x", ChannelKind.VOLTAGE),
             _ch(1, "current", ChannelKind.CURRENT),
             _ch(2, "current", ChannelKind.CURRENT))
    assert resolve(twins, 1, "current", "current") == (1, None)   # the index still matches
    idx, why = resolve(twins, 0, "current", "current")
    assert idx is None and why and "not decidable" in why


def test_name_and_kind_together_not_either_alone():
    """A name alone cannot tell a renamed current channel from the voltage channel that
    took its index; a kind alone cannot tell two current channels apart."""
    resolve = _api("resolve_channel")
    renamed = (_ch(0, "voltage", ChannelKind.VOLTAGE), _ch(1, "current", ChannelKind.VOLTAGE))
    assert resolve(renamed, 1, "current", "current")[0] is None, "same name, wrong kind"
    two_currents = (_ch(0, "IN0", ChannelKind.CURRENT), _ch(1, "IN15", ChannelKind.CURRENT))
    assert resolve(two_currents, 0, "IN15", "current") == (
        1, "'IN15' (current) is now channel 1, not channel 0")


# ================================================================================================
# the sidecar — the silent path
# ================================================================================================

def _sidecar_for(tmp_path: Path) -> Path:
    data = tmp_path / "rec.abf"
    data.write_bytes(b"ABF ")
    calib = CalibrationSet()
    calib.set("d", 0, CalibrationSet.default_for(VOLT_FIRST[0]))
    calib.set("d", 1, ChannelCalibration(gain=0.5, offset=-1.0, target_units="pA",
                                         note="cal on CURRENT"))
    assert save_sidecar(data, "d", calib, VOLT_FIRST) == C.SIDECAR_WRITTEN
    return data


def test_the_sidecar_records_which_channel_it_was_measured_on(tmp_path):
    data = _sidecar_for(tmp_path)
    blob = json.loads(sidecar_path(data).read_text())
    assert blob["version"] == _api("SIDECAR_VERSION") == 2
    entry = blob["channels"]["1"]
    assert entry["name"] == "current" and entry["kind"] == "current"


def test_a_reordered_export_no_longer_calibrates_the_voltage_trace(tmp_path):
    """THE MEASURED FAILURE, on the path with no project and no banner. Before: gain 0.5 /
    'pA' / 'cal on CURRENT' installed on the voltage channel with ``issues == ()``."""
    data = _sidecar_for(tmp_path)
    calib = CalibrationSet()
    issues = load_sidecar(data, "d", calib, CURR_FIRST)

    assert calib.get("d", 0).gain == 0.5, "the calibration follows its own channel"
    assert calib.get("d", 0).note == "cal on CURRENT"
    assert calib.get("d", 1).gain == 1.0, "the voltage channel keeps its default"
    assert issues and "applied to channel 0" in issues[0], issues


def test_a_dropped_channel_leaves_its_calibration_unapplied_and_says_so(tmp_path):
    data = _sidecar_for(tmp_path)
    calib = CalibrationSet()
    issues = load_sidecar(data, "d", calib, (_ch(0, "voltage", ChannelKind.VOLTAGE),))
    assert calib.get("d", 0).gain == 1.0
    assert issues and "was NOT applied" in issues[0] and "current" in issues[0]


def test_a_version_one_sidecar_still_loads_and_stays_silent(tmp_path):
    """CONTROL / PASSES BOTH. The owner has ``.npcal.json`` files on disk written before
    this existed. They carry no identity, so they resolve by index exactly as they always
    did — and raise nothing. Written with NO new API so the identical assertions run against
    the pre-fix package."""
    data = tmp_path / "old.abf"
    data.write_bytes(b"ABF ")
    sidecar_path(data).write_text(json.dumps({"version": 1, "channels": {
        "1": {"gain": 0.25, "offset": 0.0, "target_units": "pA", "note": "keep me"}}}))
    calib = CalibrationSet()
    issues = load_sidecar(data, "d", calib, VOLT_FIRST)
    assert issues == []
    assert calib.get("d", 1).gain == 0.25 and calib.get("d", 1).note == "keep me"


def test_a_version_one_sidecar_naming_a_channel_that_does_not_exist_is_still_refused(tmp_path):
    """CONTROL / PASSES BOTH. C8's guard has to survive the version widening — it was an
    index-range check and is now expressed through the resolver, and it must still fire."""
    data = tmp_path / "bad.abf"
    data.write_bytes(b"ABF ")
    sidecar_path(data).write_text(json.dumps({"version": 1, "channels": {
        "99": {"gain": 0.25, "offset": 0.0}}}))
    calib = CalibrationSet()
    issues = load_sidecar(data, "d", calib, VOLT_FIRST)
    assert issues and "no channel 99" in issues[0]
    assert ("d", 99) not in calib.entries


def test_a_future_sidecar_version_is_still_refused(tmp_path):
    """CONTROL / PASSES BOTH — widening the accepted range must not accept everything."""
    data = tmp_path / "future.abf"
    data.write_bytes(b"ABF ")
    sidecar_path(data).write_text(json.dumps({"version": 99, "channels": {}}))
    issues = load_sidecar(data, "d", CalibrationSet(), VOLT_FIRST)
    assert issues and "unsupported version" in issues[0]


def test_a_caller_that_does_not_know_the_channels_keeps_the_old_behaviour(tmp_path):
    """CONTROL / PASSES BOTH. ``channels=()`` means "I genuinely do not know", and inventing
    a refusal there would break the loader's other callers rather than protect anybody."""
    data = _sidecar_for(tmp_path)
    calib = CalibrationSet()
    assert load_sidecar(data, "d", calib) == []
    assert calib.get("d", 1).gain == 0.5


# ================================================================================================
# the project records — v7, additive on read
# ================================================================================================

def test_all_four_record_types_carry_the_identity():
    """One record type remembering is worse than none: the fix has to cover calibrations,
    layers, results AND fits, or the same misattribution walks in through whichever was
    left out."""
    import dataclasses

    from nanoscope.core.project import CalibrationRec, FitRec, LayerRec, ResultRec

    _records_carry_identity()


def test_the_identity_survives_a_project_round_trip(tmp_path):
    _records_carry_identity()
    from nanoscope.core.project import (
        CalibrationRec, FitRec, LayerRec, ProjectState, ResultRec, load_project, save_project,
    )

    state = ProjectState(
        calibrations=[CalibrationRec("/d", 1, 0.5, -1.0, "pA", "unknown", "n",
                                     channel_name="current", channel_kind="current")],
        layers=[LayerRec("/d", 1, "#fff", True, 0.0, False,
                         channel_name="current", channel_kind="current")],
        results=[ResultRec("/d", 1, "detect.schmitt", pd.DataFrame({"event_id": [0]}),
                           channel_name="current", channel_kind="current")],
        fits=[FitRec("/d", 1, "fit.dwell", curve={"model": "exp"},
                     channel_name="current", channel_kind="current")],
    )
    back = load_project(save_project(tmp_path / "p.nanoscope", state))
    for rec in (back.calibrations[0], back.layers[0], back.results[0], back.fits[0]):
        assert (rec.channel_name, rec.channel_kind) == ("current", "current")


def test_an_unrecorded_identity_is_absent_from_the_file_never_written_as_a_guess(tmp_path):
    """The three-state spelling ``note`` needed, for the same reason. A record loaded from a
    pre-v7 project does not know which channel it belonged to, and writing a name for it
    would turn "I don't know" into a confident claim on the next ordinary re-save."""
    _records_carry_identity()
    from nanoscope.core.project import (
        CalibrationRec, ProjectState, load_project, save_project,
    )

    state = ProjectState(calibrations=[CalibrationRec("/d", 1, 0.5, 0.0, "pA", "unknown")])
    p = save_project(tmp_path / "p.nanoscope", state)
    with zipfile.ZipFile(p) as zf:
        entry = json.loads(zf.read("manifest.json"))["calibrations"][0]
    assert "channel_name" not in entry and "channel_kind" not in entry, entry
    rec = load_project(p).calibrations[0]
    assert rec.channel_name is None and rec.channel_kind is None


def test_the_version_bumped_and_min_read_version_did_not(tmp_path):
    from nanoscope.core.project import MIN_READ_VERSION, VERSION

    assert VERSION >= 7, (
        "a project carrying channel identity must not claim a version that predates it")
    assert MIN_READ_VERSION == 1


@pytest.mark.parametrize("version", [1, 3, 6])
def test_an_older_project_still_loads_and_resolves_by_index(tmp_path, version):
    """Hand-downgraded exactly as the suite's other back-compat tests do it: rewrite the
    manifest's version and strip the keys that version never had."""
    _records_carry_identity()
    from nanoscope.core.project import (
        CalibrationRec, MANIFEST_NAME, ProjectState, load_project, save_project,
    )

    src = save_project(tmp_path / "src.nanoscope", ProjectState(
        calibrations=[CalibrationRec("/d", 1, 0.5, -1.0, "pA", "unknown", "n",
                                     channel_name="current", channel_kind="current")]))
    with zipfile.ZipFile(src) as zf:
        manifest = json.loads(zf.read(MANIFEST_NAME))
    manifest["version"] = version
    for entry in manifest["calibrations"]:
        entry.pop("channel_name", None)
        entry.pop("channel_kind", None)
        if version < 5:
            entry.pop("note", None)
    old = tmp_path / f"v{version}.nanoscope"
    with zipfile.ZipFile(old, "w") as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest))
    back = load_project(old)
    rec = back.calibrations[0]
    assert rec.channel == 1 and rec.gain == 0.5
    assert rec.channel_name is None, "absence must read as absence, never as a guess"
    assert _api("resolve_channel")(VOLT_FIRST, rec.channel,
                                   rec.channel_name, rec.channel_kind) == (1, None)
