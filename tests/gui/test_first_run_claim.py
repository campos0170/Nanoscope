"""A brand-new settings store starts in Essential and gets the Guided first analysis once.

``claim_first_run`` decides it, ``gui.app.main`` is the only caller, and everything that builds a
bare ``MainWindow()`` or ``DetectionPanel()`` (a script, a probe, a test, and every existing
user's launch) must come out exactly as it did before. The tests below hold both halves: the
claim's rule on throwaway stores, and the whole process through the real ``main()``.

The claim has two steps and so three states: absent (no claim), ``pending`` (claimed before the
window is built, the guide not yet opened) and ``essential`` (the guide has opened). Each has its
tests below.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
import pytest
from PySide6 import QtCore, QtWidgets

from nanoscope.gui.first_analysis_guide import (
    CONTROL_LEVEL_CHOICE_KEY, EXISTING_CONTROL_LEVEL_KEY, FirstAnalysisGuide, claim_first_run,
    finish_first_run,
)
from nanoscope.gui.first_run_state import (
    FIRST_RUN_DONE, FIRST_RUN_KEY, FIRST_RUN_PENDING, first_run_control_level, first_run_guide_due,
)
from nanoscope.gui.panels import DetectionPanel

_TREE = Path(nanoscope.__file__).resolve().parents[1]
#: Keys an earlier build writes at the moment of a gesture, long before ``geometry`` (which only a
#: clean quit writes): a child window's place, the plot templates, the curation options. A store
#: that holds only one of these belongs to someone who has used the app.
_EARLIER_BUILD_KEYS = ("windows/about/geometry", "plots/templates", "curation/advance_after_decision")
#: One key of each kind a store can hold that is not brand new.
_NOT_BRAND_NEW = ("geometry", EXISTING_CONTROL_LEVEL_KEY, CONTROL_LEVEL_CHOICE_KEY, FIRST_RUN_KEY,
                  *_EARLIER_BUILD_KEYS)


def _store(folder: Path, name: str = "prefs.ini") -> QtCore.QSettings:
    return QtCore.QSettings(str(folder / name), QtCore.QSettings.IniFormat)


@pytest.fixture
def suite_store(qapp):
    """A default store of this file's own, and empty. The application name is switched for the test
    (as ``_inspector_hosting.private_settings`` does), so a plain ``QSettings()`` in a
    ``DetectionPanel`` or a ``MainWindow`` reads it and the suite's shared store is never touched.
    The brand-new rule asks for no key at all, which the shared store, written by every window a
    test closes, never has."""
    org, name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("FirstRunClaim")
    QtCore.QSettings().clear()
    try:
        yield QtCore.QSettings()
    finally:
        QtCore.QSettings().clear()
        qapp.setOrganizationName(org)
        qapp.setApplicationName(name)


# ------------------------------------------------------------------------------------------------
# The claim: absent -> pending
# ------------------------------------------------------------------------------------------------
def test_a_brand_new_store_is_claimed_as_pending_and_written_to_the_file_at_once(tmp_path):
    settings = _store(tmp_path)
    assert not first_run_guide_due(settings)
    assert claim_first_run(settings) is True
    assert settings.value(FIRST_RUN_KEY) == FIRST_RUN_PENDING == "pending"
    assert first_run_guide_due(settings)
    # Written to the file before it returns, not left in memory for a quit that may never come.
    assert "first_run_v1=pending" in (tmp_path / "prefs.ini").read_text()
    assert set(settings.allKeys()) == {FIRST_RUN_KEY}, "the claim writes nothing but its own key"


def test_a_pending_claim_is_owed_again_until_the_guide_has_opened(tmp_path):
    """A first launch that died before its guide opened (a crash while the window was built, a
    force-quit of a slow start, a window closed at once) must not cost the user the guide."""
    assert claim_first_run(_store(tmp_path)) is True
    for _later_launch in range(2):
        again = _store(tmp_path)
        assert claim_first_run(again) is True, "the next launch still owes the guide"
        assert again.value(FIRST_RUN_KEY) == FIRST_RUN_PENDING
        assert set(again.allKeys()) == {FIRST_RUN_KEY}, "and writes nothing for it"


def test_finishing_the_claim_makes_it_final_and_only_ever_touches_a_pending_one(tmp_path):
    settings = _store(tmp_path)
    assert claim_first_run(settings) is True
    finish_first_run(settings)
    assert settings.value(FIRST_RUN_KEY) == FIRST_RUN_DONE == "essential"
    assert "first_run_v1=essential" in (tmp_path / "prefs.ini").read_text()
    assert not first_run_guide_due(settings)
    assert claim_first_run(settings) is False
    assert claim_first_run(_store(tmp_path)) is False, "a later launch is not a first run"
    finish_first_run(settings)
    assert settings.value(FIRST_RUN_KEY) == FIRST_RUN_DONE, "finishing twice changes nothing"

    # Nothing is written for a store that holds no pending claim: an existing user's store ...
    existing = _store(tmp_path, "existing.ini")
    existing.setValue("lastDir", "/somewhere")
    existing.sync()
    finish_first_run(existing)
    assert set(existing.allKeys()) == {"lastDir"}
    # ... or one that holds nothing at all.
    empty = _store(tmp_path, "empty.ini")
    finish_first_run(empty)
    assert empty.allKeys() == []


@pytest.mark.parametrize("key", _NOT_BRAND_NEW)
def test_a_store_holding_any_one_key_is_not_brand_new(tmp_path, key):
    settings = _store(tmp_path)
    settings.setValue(key, "essential")
    settings.sync()
    assert claim_first_run(settings) is False
    assert set(settings.allKeys()) == {key}, "an existing store must not be written to"


def test_keys_shared_with_other_applications_do_not_make_a_store_not_brand_new(qapp):
    """On macOS ``allKeys()`` includes the keys of the all-applications domain unless fallbacks are
    off, so a store that holds none of its own would never look empty. The same fallback exists on
    every platform between an application's store and its organisation's, which is what this uses."""
    ini, user = QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope
    shared = QtCore.QSettings(ini, user, "FirstRunClaimOrg")
    mine = QtCore.QSettings(ini, user, "FirstRunClaimOrg", "FirstRunClaimApp")
    try:
        shared.setValue("shared/by_every_application", 1)
        shared.sync()
        assert "shared/by_every_application" in mine.allKeys(), (
            "precondition: with fallbacks on, the organisation's keys show in the application's")
        assert claim_first_run(mine) is True
        assert set(mine.allKeys()) == {FIRST_RUN_KEY}
    finally:
        mine.clear()
        shared.clear()
        mine.sync()
        shared.sync()


@pytest.mark.parametrize("damaged, surviving_keys", [
    ("[General\n", []),                                   # nothing parseable survives
    ("[General\nlastDir=/somewhere\n", ["lastDir"]),       # a key that parsed before the damage does
], ids=["no key survives", "a key survives"])
def test_a_damaged_store_is_not_brand_new(tmp_path, damaged, surviving_keys):
    """A damaged store has lost whatever keys it had (a quit's ``geometry`` among them), so it can
    hold none and still be an existing user's: when nothing survives only the damage check can say
    so. It is not a first run.

    Qt parses a file once per process and answers every later handle from a cache that has
    forgotten the damage, so the first handle here stands for whatever touches the preferences
    before the launch asks, and the claim is made through the second."""
    (tmp_path / "broken.ini").write_text(damaged)
    early = _store(tmp_path, "broken.ini")
    assert early.status() == QtCore.QSettings.Status.FormatError, (
        "precondition: Qt itself must call this file damaged, or the test proves nothing")
    del early
    settings = _store(tmp_path, "broken.ini")
    assert settings.status() == QtCore.QSettings.Status.NoError, (
        "precondition: the cache hides the damage from a second handle")
    assert settings.allKeys() == surviving_keys, "precondition: what the damage left behind"
    assert claim_first_run(settings) is False
    assert not settings.contains(FIRST_RUN_KEY)


def test_a_claim_that_cannot_be_kept_is_withdrawn(tmp_path):
    """A first run that cannot be remembered would repeat on every launch."""
    locked = tmp_path / "locked"
    locked.mkdir()
    settings = _store(locked)
    locked.chmod(0o500)
    try:
        assert claim_first_run(settings) is False
        assert not settings.contains(FIRST_RUN_KEY)
    finally:
        locked.chmod(0o700)


# ------------------------------------------------------------------------------------------------
# What the claim does to the panel and the guide
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("value, level", [
    (None, "advanced"), (FIRST_RUN_PENDING, "essential"), (FIRST_RUN_DONE, "essential"),
    ("advanced", "advanced"), ("something else", "advanced")])
def test_the_claim_stands_for_essential_only_while_it_is_pending_or_final(tmp_path, value, level):
    settings = _store(tmp_path)
    if value is not None:
        settings.setValue(FIRST_RUN_KEY, value)
    assert first_run_control_level(settings, "advanced") == level
    assert first_run_control_level(settings, "a default") == (
        "essential" if level == "essential" else "a default")


def test_the_panel_reads_the_claim_last_and_still_defaults_to_advanced(qtbot, suite_store):
    def level() -> str:
        panel = DetectionPanel()
        qtbot.addWidget(panel)
        return panel.control_level()

    assert level() == "advanced", "the panel's empty-store default is unchanged"
    assert claim_first_run(suite_store) is True
    assert suite_store.value(FIRST_RUN_KEY) == FIRST_RUN_PENDING
    assert level() == "essential", "a pending claim starts the panel in Essential"
    finish_first_run(suite_store)
    assert level() == "essential", "and so does the final one"
    suite_store.setValue(CONTROL_LEVEL_CHOICE_KEY, "advanced")
    assert level() == "advanced", "the guide's first-use choice beats the claim"
    suite_store.remove(CONTROL_LEVEL_CHOICE_KEY)
    suite_store.setValue(EXISTING_CONTROL_LEVEL_KEY, "advanced")
    assert level() == "advanced", "the selector's own saved choice beats the claim"
    suite_store.setValue(EXISTING_CONTROL_LEVEL_KEY, "essential")
    suite_store.setValue(CONTROL_LEVEL_CHOICE_KEY, "advanced")
    assert level() == "essential", "and beats the first-use choice too, as it always did"


def test_the_claim_leaves_the_guides_choice_box_and_its_route_to_advanced(qtbot, suite_store):
    """Why the claim has a key of its own: writing either control-level key would hide the
    "Start with Advanced" button an expert on a new machine needs."""
    assert claim_first_run(suite_store) is True
    guide = FirstAnalysisGuide()
    qtbot.addWidget(guide)
    guide.show()
    guide.refresh_context()
    assert not guide.first_use_choice.isHidden()
    panel = DetectionPanel()
    qtbot.addWidget(panel)
    guide.controlLevelRequested.connect(lambda level: panel.set_control_level(level, persist=True))
    assert panel.control_level() == "essential"
    guide.advanced_choice.click()
    assert panel.control_level() == "advanced"
    assert suite_store.value(EXISTING_CONTROL_LEVEL_KEY) == "advanced"


def test_a_bare_main_window_is_never_a_first_run(qtbot, suite_store):
    """The claim is main()'s, not the window's: a script, a probe or a test that builds a
    MainWindow on an empty store gets Advanced, no guide and no claim written."""
    from nanoscope.gui.main_window import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._analysis_panel.detection_panel.control_level() == "advanced"
    assert window._first_analysis_guide is None
    assert not suite_store.contains(FIRST_RUN_KEY)
    window._analysis_controller.abort()
    window._teardown()


# ------------------------------------------------------------------------------------------------
# The guide waits for a modal question, is never opened over a closed window, and the claim is final
# only once the guide has opened
# ------------------------------------------------------------------------------------------------
def _main_with(qapp, monkeypatch, window_class, run_loop) -> int:
    """The real ``gui.app.main`` with a stand-in window class (a QObject: ``main()`` hangs a timer on
    it) and what it needs of Qt stubbed. The real claim runs, on the store the caller has made."""
    import nanoscope.gui.app as app_module
    import nanoscope.gui.main_window as main_window
    import nanoscope.gui.worker_thread as worker_thread

    monkeypatch.setattr(worker_thread, "drain_detached", lambda *_a, **_k: 0)
    monkeypatch.setattr(main_window, "MainWindow", window_class)
    monkeypatch.setattr(app_module, "create_app", lambda argv: qapp)
    monkeypatch.setattr(app_module, "_apply_theme", lambda app: None)
    monkeypatch.setattr(app_module, "configure_pyqtgraph", lambda: None)
    monkeypatch.setattr(app_module, "_FIRST_RUN_GUIDE_RETRY_MS", 20)
    monkeypatch.setattr(QtWidgets.QApplication, "exec", lambda self: run_loop())
    return app_module.main([])


def _window_with_a_modal_question(windows: list, opened: list, claim_store, base=QtCore.QObject):
    """A stand-in for the main window that has the crash-recovery question up when it shows, and
    notes (question still visible?, the claim at that moment) whenever the guide is opened. The
    question is a real, shown, application-modal dialog. ``base`` is what the stand-in is: a QObject
    (all ``main()`` needs of it is a timer's context) or a QWidget, which ``show()`` shows as well,
    as the real window is one."""

    class _Window(base):
        def __init__(self, **_kwargs):
            super().__init__()
            self.question = QtWidgets.QDialog()
            self.question.setModal(True)
            windows.append(self)

        def show(self):
            if isinstance(self, QtWidgets.QWidget):
                super().show()
            self.question.show()

        def _open_first_analysis_guide(self):
            opened.append((self.question.isVisible(), claim_store.value(FIRST_RUN_KEY)))

    return _Window


def test_the_first_run_guide_waits_for_a_modal_question_and_opens_when_it_is_answered(
        qapp, qtbot, monkeypatch, suite_store):
    """The crash-recovery offer is a modal box on screen at the same moment (the window's
    ``showEvent`` queues it ahead of main()'s own timer, and its nested event loop lets that timer
    fire)."""
    from PySide6.QtTest import QTest

    windows: list = []
    opened: list[tuple[bool, str]] = []
    while_up: dict = {}

    def loop() -> int:
        QTest.qWait(150)                          # several retries, the question still up
        while_up["opened"] = list(opened)
        while_up["claim"] = suite_store.value(FIRST_RUN_KEY)
        assert QtWidgets.QApplication.activeModalWidget() is windows[0].question, "precondition"
        windows[0].question.close()
        qtbot.waitUntil(lambda: bool(opened), timeout=5000)   # the next retry
        QTest.qWait(60)                                       # ...and nothing opens it a second time
        return 0

    window_class = _window_with_a_modal_question(windows, opened, suite_store)
    assert _main_with(qapp, monkeypatch, window_class, loop) == 0
    assert while_up == {"opened": [], "claim": FIRST_RUN_PENDING}, (
        "the guide must not open over a modal question, and the claim stays pending")
    assert opened == [(False, FIRST_RUN_PENDING)], (
        "once the question is answered the guide opens, exactly once, and the claim is still "
        "pending while it does")
    assert suite_store.value(FIRST_RUN_KEY) == FIRST_RUN_DONE, "it is final only once the guide has opened"


def test_a_window_that_goes_while_the_question_is_up_cancels_the_wait_and_leaves_the_claim_pending(
        qapp, monkeypatch, suite_store):
    """The retry timer belongs to the window: it never reaches a window that has been destroyed, and
    the claim, still pending, leaves the guide to the next launch. (The real window is not destroyed
    when it is closed, only hidden: that is the next test.)"""
    import shiboken6
    from PySide6.QtTest import QTest

    windows: list = []
    opened: list = []

    def loop() -> int:
        QTest.qWait(80)                           # a few retries, the question up
        shiboken6.delete(windows[0])              # the window is destroyed
        windows[0].question.close()               # ...and then the question is answered
        QTest.qWait(150)
        return 0

    window_class = _window_with_a_modal_question(windows, opened, suite_store)
    assert _main_with(qapp, monkeypatch, window_class, loop) == 0
    assert opened == [], "the guide was opened for a window that no longer exists"
    assert suite_store.value(FIRST_RUN_KEY) == FIRST_RUN_PENDING
    assert claim_first_run(suite_store) is True, "so the next launch opens it"


def test_a_window_that_is_closed_while_the_question_is_up_gets_no_guide_and_leaves_the_claim_pending(
        qapp, qtbot, monkeypatch, suite_store):
    """Closed is not gone. The real main window is only hidden when it is closed (``WA_DeleteOnClose``
    is set nowhere), so the timer that belongs to it lives on and, with nothing modal left to wait
    for, would open the guide over a window that has been closed and make the claim final. The
    stand-in is a real QWidget, shown, then closed and kept; the retry must see that and stop."""
    import shiboken6
    from PySide6.QtTest import QTest

    windows: list = []
    opened: list = []

    def loop() -> int:
        window = windows[0]
        qtbot.addWidget(window)
        qtbot.addWidget(window.question)
        QTest.qWait(80)                           # a few retries, the question up
        assert window.isVisible(), "precondition: the window is on screen"
        assert QtWidgets.QApplication.activeModalWidget() is window.question, "precondition"
        window.close()                            # closed: hidden, and still there
        assert shiboken6.isValid(window) and not window.isVisible(), "precondition: closed, not deleted"
        window.question.close()                   # ...and then the question is answered
        QTest.qWait(150)                          # several more retries, nothing modal left
        assert shiboken6.isValid(window), "the window was only ever closed"
        return 0

    window_class = _window_with_a_modal_question(windows, opened, suite_store, base=QtWidgets.QWidget)
    assert _main_with(qapp, monkeypatch, window_class, loop) == 0
    assert opened == [], "the guide was opened over a window that has been closed"
    assert suite_store.value(FIRST_RUN_KEY) == FIRST_RUN_PENDING
    assert claim_first_run(suite_store) is True, "so the next launch opens it"


def test_a_window_that_is_still_on_screen_gets_its_guide_when_the_question_is_answered(
        qapp, qtbot, monkeypatch, suite_store):
    """The check for a closed window must leave an open one alone. The same wait as the first test, but
    the stand-in is a QWidget (the only kind of window the check looks at, as the real one is) and it
    stays on screen."""
    from PySide6.QtTest import QTest

    windows: list = []
    opened: list[tuple[bool, str]] = []
    seen: dict = {}

    def loop() -> int:
        window = windows[0]
        qtbot.addWidget(window)
        qtbot.addWidget(window.question)
        QTest.qWait(80)                           # several retries, the question up
        seen["opened_while_up"] = list(opened)
        seen["on_screen_while_up"] = window.isVisible()
        window.question.close()
        qtbot.waitUntil(lambda: bool(opened), timeout=5000)   # the next retry
        QTest.qWait(60)                                       # ...and nothing opens it a second time
        seen["on_screen_after"] = window.isVisible()
        return 0

    window_class = _window_with_a_modal_question(windows, opened, suite_store, base=QtWidgets.QWidget)
    assert _main_with(qapp, monkeypatch, window_class, loop) == 0
    assert seen == {"opened_while_up": [], "on_screen_while_up": True, "on_screen_after": True}, (
        "the guide must not open over the modal question, and the window stays on screen throughout")
    assert opened == [(False, FIRST_RUN_PENDING)], (
        "once the question is answered the guide opens, exactly once, and the claim is still "
        "pending while it does")
    assert suite_store.value(FIRST_RUN_KEY) == FIRST_RUN_DONE, "it is final only once the guide has opened"


# ------------------------------------------------------------------------------------------------
# The real launch path, in a whole process
# ------------------------------------------------------------------------------------------------
_MAIN_PROBE = '''
import json, os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import nanoscope
print("nanoscope <-", nanoscope.__file__)
from PySide6 import QtCore, QtWidgets
QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, QtCore.QSettings.UserScope, sys.argv[1])
from nanoscope.gui import app as APP

# What a store that is not brand new already holds when the launch starts.
SEEDS = {
    "existing": {"geometry": QtCore.QByteArray(b"a layout left by an earlier quit")},
    "earlier build": {"windows/about/geometry": QtCore.QByteArray(b"a place remembered by a session"),
                      "plots/templates": "[]"},
}
if sys.argv[2] in SEEDS:
    APP.create_app([])                # names the organisation and application, as main() will
    seed = QtCore.QSettings()
    for key, value in SEEDS[sys.argv[2]].items():
        seed.setValue(key, value)
    seed.sync()
    del seed

def fake_exec(self):
    if sys.argv[2] == "dies":
        sys.stdout.flush()
        os._exit(3)                   # a launch that ends before the guide has opened
    window = [w for w in QtWidgets.QApplication.topLevelWidgets() if type(w).__name__ == "MainWindow"][0]
    for _ in range(10):               # let the zero-delay timers main() posted run
        QtWidgets.QApplication.processEvents()
        QtCore.QThread.msleep(20)
    guide = window._first_analysis_guide
    print("RESULT " + json.dumps({
        "guide_open": guide is not None and guide.isVisible(),
        "choice_box_shown": guide is not None and not guide.first_use_choice.isHidden(),
        "level": window._analysis_panel.detection_panel.control_level(),
    }))
    window._dirty = False
    window.close()                    # the real quit path: closeEvent -> _teardown -> settings saved
    return 0

QtWidgets.QApplication.exec = fake_exec
rc = APP.main([])
sys.stdout.flush()
sys.exit(rc)
'''


def _launch(tmp_path: Path, store: Path, scenario: str) -> dict:
    script = tmp_path / "main_probe.py"
    script.write_text(_MAIN_PROBE)
    # sys.path[0] of a script is its own folder, so without PYTHONPATH the subprocess would import
    # whichever nanoscope is installed; the "nanoscope <-" line asserted below is what checks it.
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(_TREE))
    proc = subprocess.run([sys.executable, str(script), str(store), scenario], cwd=str(_TREE),
                          capture_output=True, text=True, timeout=240, env=env)
    assert str(_TREE) in proc.stdout, f"the subprocess imported another tree:\n{proc.stdout}"
    if scenario == "dies":
        assert proc.returncode == 3, proc.stderr[-2000:]
        return {}
    assert proc.returncode == 0, proc.stderr[-2000:]
    (line,) = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    return json.loads(line[len("RESULT "):])


def _the_store(store: Path) -> QtCore.QSettings:
    (path,) = store.glob("*/NanoScope.ini")
    return QtCore.QSettings(str(path), QtCore.QSettings.IniFormat)


def test_main_opens_the_guide_once_and_keeps_essential_on_a_brand_new_store(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    first = _launch(tmp_path, store, "new")
    assert first == {"guide_open": True, "choice_box_shown": True, "level": "essential"}
    kept = _the_store(store)
    assert kept.value(FIRST_RUN_KEY) == FIRST_RUN_DONE, "the guide opened, so the claim is final"
    assert kept.contains("geometry"), "the quit left the layout an existing user always has"
    assert not kept.contains(EXISTING_CONTROL_LEVEL_KEY) and not kept.contains(CONTROL_LEVEL_CHOICE_KEY)
    # The next launch of the same store: no guide, and the view does not flip to Advanced.
    again = _launch(tmp_path, store, "again")
    assert again == {"guide_open": False, "choice_box_shown": False, "level": "essential"}


def test_a_first_launch_that_dies_before_its_guide_opens_gets_the_guide_next_time(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    assert _launch(tmp_path, store, "dies") == {}
    died = _the_store(store)
    assert died.allKeys() == [FIRST_RUN_KEY] and died.value(FIRST_RUN_KEY) == FIRST_RUN_PENDING, (
        "the launch claimed the store and left nothing but the pending claim")
    del died
    next_launch = _launch(tmp_path, store, "after the death")
    assert next_launch == {"guide_open": True, "choice_box_shown": True, "level": "essential"}
    assert _the_store(store).value(FIRST_RUN_KEY) == FIRST_RUN_DONE
    assert _launch(tmp_path, store, "once more") == {
        "guide_open": False, "choice_box_shown": False, "level": "essential"}


@pytest.mark.parametrize("scenario", ["existing", "earlier build"])
def test_main_leaves_an_existing_users_launch_as_it_was(tmp_path, scenario):
    """``existing``: the layout a clean quit leaves. ``earlier build``: a store that session never
    quit cleanly, so it holds a child window's place and the plot templates but no ``geometry``."""
    store = tmp_path / "store"
    store.mkdir()
    launch = _launch(tmp_path, store, scenario)
    assert launch == {"guide_open": False, "choice_box_shown": False, "level": "advanced"}
    assert not _the_store(store).contains(FIRST_RUN_KEY), "an existing store is never claimed"
