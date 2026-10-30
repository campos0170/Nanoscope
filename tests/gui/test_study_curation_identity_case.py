"""One review copy under two case spellings is one source to the Acid and Manual studies.

``Path.resolve()`` canonicalises ``..`` and symlinks but does NOT fold case, and the owner's
disk (macOS's default APFS) is case-INSENSITIVE: ``…/prepared-review.h5`` and
``…/PREPARED-REVIEW.H5`` resolve to two strings that name one file (``os.path.samefile``
True). The study windows asked "is this the review copy?" by comparing such strings, and on
the tree as found, with the study's descriptor and the main workspace naming the copy by two
spellings:

* re-entering curation (Acid ``open_main_curation``, Manual ``open_main_analysis``) did not
  find the open copy, exported and opened a SECOND one (2 documents) and left the curator's
  unsaved edit behind in the first;
* the Acid snapshot refused the selected copy: "Select this acid recording's prepared review
  copy before saving its edits.";
* the Acid commit refused a snapshot of it: "This review snapshot no longer matches the acid
  study. Open the current events again.";
* a Manual commit of its detections succeeded but left the study's bookkeeping stale, so the
  curation bar then read "The study changed. Return to study and open its current events."
  with Save disabled.

Each site now asks ``core.identity.same_source``, the filesystem's answer (device and inode),
documented there as THE answer to this question. Both studies' snapshots also pass through
``study_event_bridge.snapshot_events``, which asks the same question of the same two paths
at its own sites. So the tests that need a snapshot take it under the exact spelling and
respell what comes back, or stand the native snapshot in: each pins one study-window
compare, whatever that module answers.

Synthetic data only. Skipped on a case-sensitive volume, where two spellings are two files.
"""
import os
import shutil
from pathlib import Path

import pytest

from nanoscope.gui import acid_event_bridge as bridge
from nanoscope.gui.study_event_bridge import catalog_signature
from .test_acid_main_curation import complete, opened, windows as acid_windows  # noqa: F401
from .test_manual_main_curation import windows as manual_windows  # noqa: F401


@pytest.fixture(autouse=True)
def _case_insensitive_volume(tmp_path):
    probe = tmp_path / 'CaseProbe.tmp'
    probe.write_text('x')
    try:
        if not (tmp_path / 'caseprobe.tmp').exists():
            pytest.skip('this volume is case-sensitive: the two spellings are two real files')
    finally:
        probe.unlink()


def _respelled(path):
    """The same file under another spelling: the case of every letter in its name swapped."""
    path = Path(path)
    other = path.with_name(path.name.swapcase())
    assert str(other) != str(path) and os.path.samefile(other, path), 'premise: one file, two spellings'
    return str(other)


def _snapshots_respelled(monkeypatch, owner, name):
    """Snapshots from ``owner.name`` name the review copy by its other spelling."""
    snapshot_events = getattr(owner, name)
    def respelled(*args):
        snapshot = snapshot_events(*args)
        snapshot['provenance']['source_path'] = _respelled(snapshot['provenance']['source_path'])
        return snapshot
    monkeypatch.setattr(owner, name, respelled)


def test_acid_reentry_adopts_the_open_review_copy_under_a_case_variant(qtbot, acid_windows):
    main, study, result = opened(qtbot, acid_windows)
    descriptor = study.state['records'][0]['main_workspace_bridge']
    descriptor['path'] = _respelled(descriptor['path'])
    main._analysis_panel.set_event_accepted(0, False)
    main._return_to_study()
    complete(qtbot, study, study.open_main_curation)
    assert len(main._session.documents()) == 1, 'a second review copy was exported and opened'
    assert main._analysis_panel.current_detection() is result
    assert not result.payload.df.accepted.iloc[0], "the curator's unsaved edit was left behind"


def test_manual_reentry_adopts_the_open_review_copy_under_a_case_variant(qtbot, manual_windows):
    main, study, seed = manual_windows; seed()
    complete(qtbot, study, study.open_main_analysis)
    native = main._analysis_panel.current_detection()
    descriptor = study.state['records'][0]['main_workspace_bridge']
    descriptor['path'] = _respelled(descriptor['path'])
    main._analysis_panel.set_event_accepted(0, False)
    main._return_to_study()
    complete(qtbot, study, study.open_main_analysis)
    assert len(main._session.documents()) == 1, 'a second review copy was exported and opened'
    assert main._analysis_panel.current_detection() is native
    assert not native.payload.df.accepted.iloc[0], "the curator's unsaved edit was left behind"


def test_acid_snapshot_takes_its_review_copy_under_a_case_variant(qtbot, acid_windows, monkeypatch):
    main, study, result = opened(qtbot, acid_windows)
    descriptor = study.state['records'][0]['main_workspace_bridge']
    descriptor['path'] = _respelled(descriptor['path'])
    # Only the Acid gate is under test: the native snapshot behind it asks the same
    # question of the same two paths in study_event_bridge.
    reached = []
    monkeypatch.setattr(bridge.native, 'snapshot_events',
                        lambda window, adapter: reached.append(adapter['tag']) or dict(provenance={}))
    snapshot = bridge.snapshot_events(main, study.state, 'acid_30h')
    assert reached == ['acid_30h']
    assert snapshot['provenance']['acid_recording_id'] == 'acid-record-1'


def test_acid_save_takes_a_snapshot_that_spells_its_review_copy_another_way(qtbot, acid_windows, monkeypatch):
    main, study, result = opened(qtbot, acid_windows)
    _snapshots_respelled(monkeypatch, bridge, 'snapshot_events')
    main._analysis_panel.set_event_accepted(0, False)
    complete(qtbot, study, main._save_study_curation)
    assert study.state['records'][0]['catalog'].accepted.tolist() == [False, True]


def test_manual_use_of_main_detections_keeps_its_bookkeeping_under_a_case_variant(qtbot, manual_windows, monkeypatch):
    main, study, seed = manual_windows; seed()
    complete(qtbot, study, study.open_main_analysis)
    _snapshots_respelled(monkeypatch, main, 'snapshot_study_events')
    main._analysis_panel.set_event_accepted(0, False)
    main._return_to_study()
    complete(qtbot, study, study.use_main_detections)
    record = study.state['records'][0]
    assert not record['analysis'].catalog.accepted.iloc[0]
    assert record['main_workspace_bridge']['catalog_signature'] == catalog_signature(record)
    label = main._study_curation_label.text()
    assert 'Curation saved to study' in label, label
    assert main._study_curation_save.isEnabled()


def test_a_byte_identical_copy_of_the_review_is_still_another_file(qtbot, acid_windows, monkeypatch, tmp_path):
    """The scoping control: same bytes at another path is not the review copy.

    Identity is the filesystem's, never the contents; this passed before the fix too.
    """
    main, study, result = opened(qtbot, acid_windows)
    copy = tmp_path / 'elsewhere' / 'prepared-review.h5'
    copy.parent.mkdir()
    shutil.copyfile(study.state['records'][0]['main_workspace_bridge']['path'], copy)
    snapshot_events = bridge.snapshot_events
    def elsewhere(*args):
        snapshot = snapshot_events(*args)
        snapshot['provenance']['source_path'] = str(copy)
        return snapshot
    monkeypatch.setattr(bridge, 'snapshot_events', elsewhere)
    main._analysis_panel.set_event_accepted(0, False)
    failures = []
    study.operationFailed.connect(failures.append)
    assert main._save_study_curation()
    qtbot.waitUntil(lambda: not study.is_busy(), timeout=30000)
    assert failures and 'no longer matches the acid study' in failures[0], failures
    assert study.state['records'][0]['catalog'].accepted.tolist() == [True, True]
