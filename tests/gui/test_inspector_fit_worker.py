"""Fits stay responsive, honour cancellation and never install an obsolete source snapshot."""
import threading
import time
import pytest
from PySide6 import QtCore
from .test_fit_provenance import window, _install_and_detect
from .fit_test_helpers import wait_for_fit
from nanoscope.gui import analysis_panel as module
from nanoscope.analysis.cancel import check_cancelled


def _fast_fitters(monkeypatch):
    for name in ('fit_dwell', 'fit_depth'):
        original = getattr(module, name)
        monkeypatch.setattr(module, name,
            lambda *args, _fn=original, **kw: _fn(*args, n_boot=0, **kw))


def test_fit_runs_off_gui_thread_and_does_not_modify_events(window, qtbot, tmp_path, monkeypatch):
    doc_id, channel = _install_and_detect(window, qtbot, tmp_path)
    panel = window._analysis_panel
    source = panel.current_events().df.copy(deep=True)
    _fast_fitters(monkeypatch)
    original = module.fit_dwell
    threads = []
    release = threading.Event()
    def delayed(*args, **kwargs):
        threads.append(threading.get_ident())
        assert release.wait(5), 'GUI timer did not run while fitting'
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'fit_dwell', delayed)
    QtCore.QTimer.singleShot(50, release.set)
    panel._stats.fit_button.click()
    assert panel._stats._fit_busy
    wait_for_fit(panel._stats)
    assert threads == [threads[0]] and threads[0] != threading.get_ident()
    assert window._session.results.get(doc_id, channel, 'fit.dwell') is not None
    import pandas as pd
    pd.testing.assert_frame_equal(source, panel.current_events().df)


def test_cancel_fit_keeps_previous_results(window, qtbot, tmp_path, monkeypatch):
    doc_id, channel = _install_and_detect(window, qtbot, tmp_path)
    panel = window._analysis_panel
    _fast_fitters(monkeypatch)
    panel._stats.fit_button.click(); wait_for_fit(panel._stats)
    previous = window._session.results.get(doc_id, channel, 'fit.dwell')
    entered = threading.Event()
    def slow(*args, **kwargs):
        entered.set()
        for _ in range(500):
            check_cancelled(); time.sleep(.01)
        raise AssertionError('cancel was not polled')
    monkeypatch.setattr(module, 'fit_dwell', slow)
    panel._stats.fit_button.click()
    qtbot.waitUntil(entered.is_set, timeout=2000)
    panel._stats.fit_button.click()
    wait_for_fit(panel._stats)
    assert window._session.results.get(doc_id, channel, 'fit.dwell') is previous
    assert 'cancelled' in panel._stats.results.toPlainText().lower()


def test_event_edit_during_fit_discards_detached_result(window, qtbot, tmp_path, monkeypatch):
    doc_id, channel = _install_and_detect(window, qtbot, tmp_path)
    panel = window._analysis_panel
    _fast_fitters(monkeypatch)
    original = module.fit_dwell
    release, entered = threading.Event(), threading.Event()
    def delayed(*args, **kwargs):
        entered.set(); assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'fit_dwell', delayed)
    panel._stats.fit_button.click()
    qtbot.waitUntil(entered.is_set, timeout=2000)
    df = panel.current_events().df
    df.loc[df.index[0], 'accepted'] = not bool(df.iloc[0]['accepted'])
    release.set(); wait_for_fit(panel._stats)
    assert window._session.results.get(doc_id, channel, 'fit.dwell') is None
    assert 'changed during the run' in panel._stats.results.toPlainText()


def test_provenance_change_during_fit_discards_detached_result(window, qtbot, tmp_path, monkeypatch):
    doc_id, channel = _install_and_detect(window, qtbot, tmp_path)
    panel = window._analysis_panel
    _fast_fitters(monkeypatch)
    original = module.fit_dwell
    release, entered = threading.Event(), threading.Event()
    def delayed(*args, **kwargs):
        entered.set(); assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'fit_dwell', delayed)
    panel._stats.fit_button.click()
    qtbot.waitUntil(entered.is_set, timeout=2000)
    panel.current_events().provenance['min_dwell_s'] *= 2
    release.set(); wait_for_fit(panel._stats)
    assert window._session.results.get(doc_id, channel, 'fit.dwell') is None
    assert 'changed during the run' in panel._stats.results.toPlainText()
