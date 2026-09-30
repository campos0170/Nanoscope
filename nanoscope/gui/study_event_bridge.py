"""File-backed, explicitly committed exchange with the main analysis workspace."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..analysis.baseline import BaselineConfig, characterisation_baseline
from ..analysis.cancel import check_cancelled, propagate_cancel, resolve_cancel
from ..analysis.characterize import characterize
from ..analysis.events import (EventSet, REVIEW_NONE, analysed_span,
                               span_provenance, stamp_original_bounds, stamp_source_status,
                               DERIVED_FLAGS, flag_tokens, combine_flags,
                               STUDY_REVIEW_POLICY_COL, STUDY_REVIEW_POLICY)
from ..analysis.incubation_detection import _fingerprint
from ..analysis.manual_detection import recording_identity, _array_hash
from ..analysis.study_review import event_reviews, event_review_conflicts
from ..core.analysis import AnalysisResult
from ..core.calibration import (CalibrationSet, ChannelCalibration, save_sidecar,
                                SIDECAR_WRITTEN)
from ..core.document import TraceDocument
from ..core.identity import fingerprint_source, describe_source_change
from ..core.segment import SegmentDocument
from ..core.types import Provenance, STUDY_BRIDGE_ATTR_KEY as BRIDGE_KEY
from ..io.hdf5 import Hdf5Reader

MODULE_ID = "detect.study"
SCHMITT_SOURCE_ID = "detect.study.schmitt"
BRIDGE_VERSION = 1


def _catalog(record):
    analysis = record.get("analysis")
    return analysis.catalog if analysis is not None else record.get(
        "candidates", pd.DataFrame(columns=["start_idx", "end_idx"]))


def catalog_signature(record):
    """Anchor the exchange to the committed event table, including review decisions."""
    frame = _catalog(record)
    frame = frame.copy(deep=False)
    for name in frame.select_dtypes(include="object"):
        frame[name] = frame[name].map(lambda value: json.dumps(value, sort_keys=True, default=str)
            if isinstance(value, (dict, list, tuple, np.ndarray)) else value)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


@propagate_cancel
def prepare_review_export(record: dict, output_folder: Path, *, cancel=None) -> dict:
    """Export an independent prepared trace and native curation table.

    The native baseline is a constant median of the prepared current, and native
    columns retain their ordinary definitions. Original study measurements are
    carried under ``study_*`` names and are remeasured by the study on import.
    """
    prepared = record.get("prepared")
    if prepared is None:
        raise ValueError("Prepare the selected recording before opening it in the main workspace.")
    identity = recording_identity(prepared, cancel=cancel)
    recording_id = str(record.get("recording_id") or "")
    if not recording_id:
        raise ValueError("The study recording has no stable identity. Reopen the study first.")
    folder = Path(output_folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "prepared-review.h5"
    if path.exists():
        raise FileExistsError("Review export already exists; use a new review stage.")
    fs, signal = float(prepared.fs_hz), np.asarray(prepared.detrended, np.float32)
    # HDF5's normal reader verifies the entire clock and can reopen this file in
    # projects. The original ABF is never used as a synthetic document's path.
    with h5py.File(path, "x") as stream:
        current = stream.create_dataset("current", shape=signal.shape, dtype="f4")
        clock = stream.create_dataset("time", shape=signal.shape, dtype="f8")
        for start in range(0, len(signal), 262144):
            check_cancelled(cancel)
            stop = min(start + 262144, len(signal))
            current[start:stop] = signal[start:stop]
            clock[start:stop] = np.arange(start, stop, dtype=np.float64) / fs
        stream.attrs[BRIDGE_KEY] = json.dumps(dict(version=BRIDGE_VERSION,
            recording_id=recording_id, recording_identity=identity, units="pA",
            orientation="events_down"), sort_keys=True)
    check_cancelled(cancel)
    document = Hdf5Reader(path).load(cancel=resolve_cancel(cancel))
    if not np.array_equal(document.raw(0), signal):
        raise ValueError("Prepared review export did not preserve the current samples.")
    # Use the reader's actual floating clock for every native measurement. A
    # one-ULP difference from the requested clock must not make a later native
    # remeasurement appear to move t_start_s. The study retains its own exact fs.
    fs = float(document.meta.sample_rate_hz)
    # HDF5's generic reader does not infer physical units. An ordinary calibration
    # sidecar declares them without applying a gain; normal project reload uses it.
    calibration = CalibrationSet()
    calibration.set(document.doc_id, 0, ChannelCalibration(target_units="pA",
        provenance=Provenance.FILE_META, note="Prepared study export; physical current in pA."))
    reason = []
    if save_sidecar(path, document.doc_id, calibration, document.meta.channels, reason) != SIDECAR_WRITTEN:
        raise ValueError("Could not save review export units: " + "; ".join(reason))
    descriptor = dict(version=BRIDGE_VERSION, recording_id=recording_id,
        recording_identity=identity, path=str(path), channel=0,
        file_sha256=_fingerprint(path, cancel)["sha256"],
        catalog_signature=catalog_signature(record))
    frame = _catalog(record)
    starts = frame.start_idx.to_numpy(np.int64)
    ends = frame.end_idx.to_numpy(np.int64) + 1
    baseline = characterisation_baseline(signal, fs,
        config=BaselineConfig(method="constant"), polarity=-1, health=False, cache=False)
    noise = max(float(np.median(np.abs(signal - np.median(signal))) * 1.4826), 1e-12)
    native = characterize(signal, baseline.baseline, starts, ends, fs, -1,
        noise_rms=noise, pore_id=str(record.get("pore_id") or ""),
        recording=str(record.get("tag") or prepared.tag), units="pA", cancel=cancel)
    if "accepted" in frame:
        native["accepted"] = frame.accepted.to_numpy(bool)
    analysis = record.get("analysis")
    source_id = (SCHMITT_SOURCE_ID if analysis is not None and "schmitt_settings" in analysis.provenance
                 else MODULE_ID)
    native = stamp_source_status(native, source_id, review=REVIEW_NONE)
    native = stamp_original_bounds(native)
    # A return trip must carry the human's work back into the active native
    # columns. The container's new name does not turn reviewed events into new
    # machine candidates, or erase the detector/manual identity of each row.
    if "source_event_id" in frame:
        ids = frame.source_event_id.to_numpy()
        numeric = np.asarray(ids, float)
        if (not np.isfinite(numeric).all() or np.any(numeric != np.floor(numeric))
                or len(np.unique(numeric)) != len(numeric)):
            raise ValueError("Imported source event IDs must remain unique integer identities.")
        native["event_id"] = np.asarray(ids, np.int64)
        for name in ("source", "added", "orig_start_idx", "orig_end_idx"):
            if "source_" + name in frame:
                native[name] = frame["source_" + name].to_numpy(copy=True)
        if "source_flag" in frame:
            native["flag"] = pd.array([
                combine_flags(fresh, ";".join(t for t in flag_tokens(prior) if t not in DERIVED_FLAGS))
                for fresh, prior in zip(native.flag, frame.source_flag)
            ], dtype="string")
    # Current study gestures (including an explicit reset) supersede archived
    # native attention. Untouched imported rows retain their recorded review.
    if analysis is not None:
        native["review"] = event_reviews(analysis).to_numpy()
        conflicts = event_review_conflicts(analysis)
        if conflicts is not None:
            native["consensus_curation_conflict"] = conflicts.to_numpy()
    native[STUDY_REVIEW_POLICY_COL] = STUDY_REVIEW_POLICY
    for name in frame.columns:
        # Source annotations have just been restored above. Re-exporting earlier
        # source_study_* archives here would grow another prefix/copy each cycle.
        if str(name).startswith("source_"):
            continue
        native["study_" + str(name)] = frame[name].to_numpy(copy=True)
    note = ("Imported study candidates on an events-down prepared trace. Native measurements use "
            "the prepared current's constant median; study_* columns preserve the study's original "
            "measurements until edits are committed and remeasured in the study. Raw-current I0 "
            "and fractional blockage are calculated in the study; native frac_depth is unavailable "
            "for this baseline-subtracted current.")
    provenance = dict(baseline.provenance(), **span_provenance(0, len(signal)),
        detector="study_candidates", units="pA", fs=fs, n_samples=len(signal),
        duration_s=len(signal) / fs, analyzed_duration_s=len(signal) / fs,
        polarity=-1, noise_rms_pA=noise, n_events=len(native),
        excluded_duration_s=0., exclude_regions=[],
        fractional_blockage_reference_unavailable=True,
        study_fixed_baseline_fingerprint=baseline.fingerprint,
        measurement_note=note, **{BRIDGE_KEY: deepcopy(descriptor)})
    events = EventSet(native, provenance)
    result = AnalysisResult(MODULE_ID, str(BRIDGE_VERSION), document.doc_id, 0,
        {"source": "manual study workspace", "range_start": 0, "range_stop": len(signal)},
        (1., 0.), payload=events, warnings=(note,), baseline_fingerprint=baseline.fingerprint,
        source_fingerprint=fingerprint_source(path, len(signal)), source_fingerprint_recorded=True)
    check_cancelled(cancel)
    return dict(document=document, result=result, descriptor=descriptor)


def _assert_idle(window):
    panel = window._analysis_panel
    if (window._has_cancellable_job() or panel._merge_job is not None
            or panel._remeasure_target is not None or panel._pending is not None):
        raise ValueError("Wait for main-workspace analysis and editing operations to finish first.")


def _assert_full_exposure(prov, n, fs):
    message = ("Run detection over the whole recording with no excluded regions before importing; "
               "study rates need full exposure.")
    # analysed_span is deliberately forgiving for legacy display, but an import
    # must not turn malformed explicitly stored ranges into full exposure.
    for left, right in (("analysed_start_idx", "analysed_stop_idx"), ("range_start", "range_stop")):
        if left in prov or right in prov:
            if left not in prov or right not in prov:
                raise ValueError(message)
            values = (prov[left], prov[right])
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in values):
                raise ValueError(message)
            if values != (0, n):
                raise ValueError(message)
    def number(value):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        if not np.isfinite(result):
            raise ValueError(message)
        return result
    if (analysed_span(prov, n) != (0, n)
            or not np.isclose(number(prov.get("fs", prov.get("fs_hz", fs))), fs, rtol=1e-12, atol=0)
            or not np.isclose(number(prov.get("analyzed_duration_s")), n / fs, rtol=1e-12, atol=0)
            or number(prov.get("excluded_duration_s", 0)) != 0
            or prov.get("exclude_regions")):
        raise ValueError(message)


def open_review(window, payload, launch_doc_ids=None):
    """Install the exported document without replacing another recording's results."""
    _assert_idle(window)
    if launch_doc_ids is not None:
        current_doc_ids = frozenset(doc.doc_id for doc in window._session.documents())
        if not launch_doc_ids <= current_doc_ids:
            raise ValueError(
                "The main workspace's project changed while this review copy was "
                "being prepared. Reopen the recording from the manual study to "
                "import it into the current project.")
    document, result = payload["document"], payload["result"]
    if window._session.doc_id_for_source(document.source_key) is not None:
        raise ValueError("This review export is already open. Continue curating its existing table.")
    window._on_loaded(document)
    if not window._session.has_document(document.doc_id):
        raise ValueError("Could not open the prepared study recording in the main workspace.")
    window._on_selection(document.doc_id, 0)
    window._session.add_result(result)
    window.show()
    window.raise_()
    window.activateWindow()
    window._open_event_gallery()
    window.statusBar().showMessage(
        "Curate these detections with the table, Event Gallery and Inspector. Return to the "
        "manual study and choose Use main workspace detections to commit them.", 20000)
    return document.doc_id


def snapshot_events(window, record):
    """Snapshot the active full-exposure result; the study owns the later commit."""
    _assert_idle(window)
    result = window._analysis_panel.current_detection()
    if result is None or not isinstance(result.payload, EventSet):
        raise ValueError("Select a detection result in the main workspace first.")
    session = window._session
    doc = session.document(result.doc_id)
    prepared = record.get("prepared")
    if prepared is None:
        raise ValueError("Prepare this study recording before importing main-workspace detections.")
    identity = recording_identity(prepared)
    n, fs = identity["n_samples"], identity["fs_hz"]
    if (not isinstance(doc, TraceDocument) or isinstance(doc, SegmentDocument)
            or doc.meta.is_partial or doc.meta.n_samples != n
            or doc.meta.sample_rate_hz is None
            or not np.isclose(doc.meta.sample_rate_hz, fs, rtol=1e-12, atol=0)):
        raise ValueError("Use the matching complete recording; partial loads and segments cannot supply study exposure.")
    cal = session.calibration(result.doc_id, result.channel)
    if cal.fingerprint() != (1., 0.) or tuple(result.calibration_fingerprint) != cal.fingerprint():
        raise ValueError("Restore the recording's original calibration and recompute stale detections before importing.")
    if session.calibrated(result.doc_id, result.channel).units != "pA":
        raise ValueError("Study detections require a source channel calibrated in pA.")
    prov = result.payload.provenance or {}
    # acid_event_bridge.snapshot_events requires its main-workspace import to have
    # polarity == -1 because the acid engine's prepared trace is always oriented
    # events-down; this engine's PreparedRecording.detrended is ALSO always oriented
    # events-down (manual_preparation.prepare_raw flips native-upward recordings), but
    # unlike acid this engine lets the user declare either native orientation per
    # record (event_polarity), so the main-workspace detector's OWN polarity setting
    # (describing the direction of a deviation in the UNFLIPPED, natively-oriented
    # document) must be the OPPOSITE of event_polarity, not a fixed -1: a recording
    # whose real events go up (event_polarity=-1) is only found by a detector run
    # with polarity=+1 on that native trace (verified against detect.detect_events:
    # a synthetic upward pulse is recovered at polarity=+1, not polarity=-1). Nothing
    # here checked this before, so a detection run with the wrong polarity for this
    # record's declared orientation imported silently, with every depth/sign
    # downstream computed as though the event went the other way.
    expected_polarity = -int(prepared.metadata.get("event_polarity", 1))
    if int(prov.get("polarity", 0)) != expected_polarity:
        raise ValueError("The main detection's polarity does not match this recording's declared "
                          "event orientation. Re-run detection with the matching polarity before importing.")
    _assert_full_exposure(prov, n, fs)
    reason = (None if result.module_id == "detect.reconciled" else
              window._analysis_panel._baseline_staleness(result))
    if reason:
        raise ValueError("The main detection measurements are stale. Re-measure or restore their baseline settings first. " + reason)
    if window._analysis_panel._reconciliation_source_staleness(result):
        raise ValueError("Reconcile the updated source detections again before importing.")
    if result.module_id == "detect.reconciled":
        for module_id in window._analysis_panel._reconciliation_input_fingerprints(result):
            source_result = session.results.get(result.doc_id, result.channel, module_id)
            if source_result is None or window._analysis_panel._baseline_staleness(source_result):
                raise ValueError("A contributing detector's baseline settings are stale. Restore or re-measure its baseline, then reconcile again before importing.")
    current_source = fingerprint_source(doc.meta.path, n)
    if result.source_fingerprint_recorded and result.source_fingerprint is None:
        raise ValueError("This restored detection has no recorded source evidence. Recompute detections on the verified recording before importing.")
    loaded_source = window._loaded_source_fingerprints.get(doc.doc_id)
    if loaded_source is None or current_source is None or describe_source_change(
            loaded_source, current_source, loaded_n_samples=n):
        raise ValueError("The loaded recording's source cannot be verified unchanged. Reload it before importing.")
    if result.source_fingerprint_recorded and describe_source_change(
            result.source_fingerprint, current_source, loaded_n_samples=n):
        raise ValueError("The main detection's source file changed; reload and recompute before importing.")
    descriptor = record.get("main_workspace_bridge")
    bridge_match = isinstance(descriptor, dict) and Path(descriptor.get("path", "")).resolve() == doc.meta.path.resolve()
    if bridge_match:
        if (descriptor.get("version") != BRIDGE_VERSION
                or descriptor.get("recording_id") != record.get("recording_id")
                or descriptor.get("recording_identity") != identity
                or descriptor.get("catalog_signature") != catalog_signature(record)
                or result.channel != descriptor.get("channel")
                or _fingerprint(doc.meta.path)["sha256"] != descriptor.get("file_sha256")
                or _array_hash(doc.raw(result.channel)) != identity["detrended_sha256"]):
            raise ValueError("The study changed since this review export. Open a fresh review copy before returning detections.")
    else:
        source = prepared.provenance.get("raw_file", {})
        if (doc.meta.path.resolve() != Path(record.get("source_path", "")).expanduser().resolve()
                or result.channel != int(record.get("channel", -1))
                or not source.get("sha256")
                or _fingerprint(doc.meta.path)["sha256"] != source["sha256"]):
            raise ValueError("Select this study recording's original raw channel or its exported prepared review copy.")
    frame = result.payload.df.copy(deep=True)
    for name in frame.select_dtypes(include="object"):
        frame[name] = frame[name].map(deepcopy)
    frame.attrs = deepcopy(result.payload.df.attrs)
    snapshot_provenance = deepcopy(prov)
    snapshot_provenance.update(source_module_id=result.module_id,
        source_module_version=result.module_version, source_path=str(doc.meta.path),
        source_channel=result.channel, source_kind="prepared_review" if bridge_match else "raw_recording",
        study_recording_id=record["recording_id"], recording_identity=identity,
        source_file_sha256=_fingerprint(doc.meta.path)["sha256"],
        study_reader_scaling=prepared.provenance.get("reader_scaling", "not recorded"),
        native_reader=doc.meta.format_name,
        source_validation=("prepared file content, loaded file fingerprint, exact prepared samples and complete sample clock"
            if bridge_match else "original file content, loaded file fingerprint, channel and complete sample clock"),
        calibration_fingerprint=list(cal.fingerprint()), index_convention="half-open [start_idx, end_idx)")
    return dict(events=frame, provenance=snapshot_provenance)
