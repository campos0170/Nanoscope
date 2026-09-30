"""Shared pytest fixtures / builders for the ported analysis + plot + gui tests.

Qt bootstrap discipline (contract §2): ``import nanoscope`` MUST run before any
Qt import so the Anaconda plugin-path fix is installed. We import it at module
top and force the offscreen platform, mirroring tests/test_integration.py.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import gc
import os

# PySide6 6.11.1 can deadlock when cyclic collection destroys a Qt wrapper.  This must
# happen before importing NanoScope (and therefore before any Qt import), and it lives in
# the in-tree harness so the documented plain ``pytest`` command cannot silently omit it.
# ``tests/lifetime/conftest.py`` deliberately enables the collector around the small set
# of real-input object-lifetime tests, then restores this disabled state.
gc.disable()

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import atexit
import hashlib
import shutil
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

# ================================================================================================
# THE SUITE MAY NOT WRITE THE OWNER'S REAL PREFERENCES. Installed at IMPORT time, not in a
# fixture, and by REDIRECTING THE STORE rather than by renaming the app.
# ================================================================================================
#
# WHAT WENT WRONG, MEASURED. The ``qapp`` fixture below renames the application
# ("NanoScopeTest" / "NanoScopeSuite") so ``QSettings`` keys somewhere private.
# ``tests/test_integration.py`` defined its OWN session fixture also called ``qapp``,
# which calls ``create_app([])`` and nothing else; a fixture in a test module SHADOWS the
# one in conftest for that module, so for every test in that file the application name was
# the shipped app's own — and ``create_app`` is what sets it back. A full run of the suite
# changed ``~/Library/Preferences/com.nanoscope-local.NanoScope.plist``
# (md5 f58fe4d1… -> faf8fd28…) and left ELEVEN pytest temporary-directory paths in it,
# ten in ``recentProjects`` and one in ``lastProjectDir`` — the store the SHIPPED APP reads
# on launch, so the owner's Recent Projects menu filled up with deleted directories.
#
# WHY RENAMING IS NOT THE FIX. A name is something the next fixture can set back, and
# ``create_app`` sets it back by design — that is its job. Sixteen test modules already
# take a private namespace by renaming and then RESTORE whatever name they found, so
# whichever fixture happens to run first decides where every later test writes. Isolation
# that any second fixture can step around is not isolation.
#
# WHAT IS DONE INSTEAD. The two statics below move the STORE ITSELF, for the whole process,
# before any fixture or test module has run:
#
#   * ``setDefaultFormat(IniFormat)`` — a plain ``QSettings()`` (the only spelling this app
#     uses) then keys on an .ini file instead of the macOS preferences domain, so cfprefsd
#     and ``~/Library/Preferences`` are out of the picture entirely;
#   * ``setPath(IniFormat, …)`` — and that .ini lives under a temporary directory.
#
# Neither depends on the organisation or application NAME, so a fixture that renames the
# app (this one, the sixteen private-namespace fixtures, or ``create_app`` itself) still
# lands inside the redirect. ``_real_settings_store_is_never_written`` below is the guard
# that FAILS if any of this is ever undone.
_SETTINGS_ROOT = Path(tempfile.mkdtemp(prefix="nanoscope-qsettings-"))
_RECOVERY_ROOT = Path(tempfile.mkdtemp(prefix="nanoscope-recovery-"))
_STUDY_LIBRARY_ROOT = Path(tempfile.mkdtemp(prefix="nanoscope-study-library-"))
# Crash-recovery snapshots are user data just like preferences.  Keep every GUI test
# inside a private store so a long-running test cannot discover, offer, overwrite or
# retire a recovery left by the installed application.
os.environ["NANOSCOPE_RECOVERY_ROOT"] = str(_RECOVERY_ROOT)
os.environ["NANOSCOPE_INCUBATION_LIBRARY_ROOT"] = str(_STUDY_LIBRARY_ROOT)
QtCore.QSettings.setDefaultFormat(QtCore.QSettings.IniFormat)
for _scope in (QtCore.QSettings.UserScope, QtCore.QSettings.SystemScope):
    QtCore.QSettings.setPath(QtCore.QSettings.IniFormat, _scope, str(_SETTINGS_ROOT))
# ...and taken away again at the end of the process. A suite that leaks a directory per
# run into /var/folders is a smaller version of the same discourtesy this block is about.
atexit.register(shutil.rmtree, str(_SETTINGS_ROOT), True)
atexit.register(shutil.rmtree, str(_RECOVERY_ROOT), True)
atexit.register(shutil.rmtree, str(_STUDY_LIBRARY_ROOT), True)


def pytest_runtest_setup(item) -> None:
    """Give each test its own default crash-recovery namespace.

    A single private directory for the whole pytest process protects the owner's real
    files, but it does not isolate tests from *one another*.  That distinction became
    observable once crash recovery was real: a long GUI test could autosave, let pytest
    destroy its window without choosing Discard, and the next test's ``MainWindow`` would
    correctly offer that checkpoint.  The later test then failed because it had never
    asked to exercise startup recovery.

    This hook runs before fixture setup (including a module-scoped fixture's first setup),
    so default ``MainWindow`` instances created by a test all stay under a deterministic
    child of the session-private root.  Tests which deliberately exercise recovery inject
    their own ``recovery_root`` and are unaffected.
    """
    token = hashlib.sha256(item.nodeid.encode("utf-8")).hexdigest()[:20]
    os.environ["NANOSCOPE_RECOVERY_ROOT"] = str(_RECOVERY_ROOT / token)
    os.environ["NANOSCOPE_INCUBATION_LIBRARY_ROOT"] = str(_STUDY_LIBRARY_ROOT / token)


def _locate_real_settings_store() -> Path:
    """Ask Qt where the SHIPPED app keys its settings — ONCE, and never again.

    Built from the org/app ``nanoscope.gui.app.create_app`` names, in the NATIVE format,
    so this resolves to the real per-platform location (a plist domain on macOS, an .ini
    under ``~/.config`` on Linux) and cannot drift from what the app actually uses.

    THE PROBE IS ITSELF A TOUCH, which is why it is called exactly once, from module
    import, above the baseline snapshot. Constructing a ``NativeFormat`` QSettings on
    macOS registers that domain with ``cfprefsd``, and destroying it runs
    ``QSettings::sync`` → ``CFPreferencesAppSynchronize`` on the owner's real domain: a
    flush the guard provoked, landing on the owner's plist, with the guard's own baseline
    taken before it. Measured against the previous shape (a fresh probe on every call,
    i.e. twice per test): a ~20 s test reported "this test wrote the owner's real settings
    store" on a run in which every one of its assertions passed and the file's CONTENT was
    byte-identical before and after — only the mtime had moved. A guard that is the thing
    that trips the guard is the one failure mode this whole block exists to prevent, and
    it also makes the honest answer unreadable: the run where a write really did land was
    indistinguishable from the runs where cfprefsd merely restamped the file.

    Hence both halves of the fix, and they are independent: this one probe instead of
    ten thousand, and :func:`real_store_signature` comparing CONTENT rather than mtime.
    """
    probe = QtCore.QSettings(QtCore.QSettings.NativeFormat, QtCore.QSettings.UserScope,
                             "nanoscope.local", "NanoScope")
    return Path(probe.fileName())


#: Resolved at import — BEFORE ``_REAL_STORE_SIG`` below, so whatever the one probe stirs up
#: is part of the baseline rather than something a test gets blamed for.
_REAL_STORE_PATH = _locate_real_settings_store()


def real_settings_store() -> Path:
    """The file the SHIPPED app reads on launch. Resolved once at import; see
    :func:`_locate_real_settings_store` for why asking twice is not free."""
    return _REAL_STORE_PATH


def real_store_signature(path: Path | None = None) -> tuple:
    """(exists, size, sha256) for the real store — its CONTENT, never its mtime.

    THE QUESTION IS "DID ANYONE WRITE THIS FILE?", NOT "DID ITS mtime MOVE?". On macOS the
    two are not the same question: ``cfprefsd`` restamps a registered domain when it
    flushes, so an mtime comparison reports a write for a file whose every byte is
    unchanged — and it was this guard's own probe that registered the domain (see
    :func:`_locate_real_settings_store`). Hashing is also the STRICTER test of the two: a
    write that happens to land on the same size and the same mtime granularity — a
    key edited in place — was invisible to the old signature and is not invisible to this
    one. Preferences files are a few kB; the read is chunked so that stays true if one day
    it is not.
    """
    target = real_settings_store() if path is None else path
    digest = hashlib.sha256()
    size = 0
    try:
        with open(target, "rb") as fh:
            while chunk := fh.read(65536):
                size += len(chunk)
                digest.update(chunk)
    except OSError:
        return (False, 0, "")
    return (True, size, digest.hexdigest())


def check_real_store_untouched(previous: tuple, path: Path | None = None) -> tuple:
    """Return the store's signature now, FAILING if it differs from ``previous``.

    A plain function rather than logic buried in the fixture, so a test can hand it a
    stand-in file and prove it actually fires — a guard that has never been seen to fail
    is indistinguishable from ``None == None``.

    THE MESSAGE NAMES BOTH CAUSES, because the guard cannot tell them apart (the fixture's
    docstring has always said so; the failure text used to offer only the first one, which
    sends the reader hunting through their own diff for a NativeFormat QSettings that was
    never there). The second cause is real and was measured DURING this fix: a packaging
    self-test in another checkout kept launching the shipped app, whose quit path writes
    ``recentFiles``, and every one of those writes was reported against whichever test
    happened to be in flight. The file's mtime is printed for exactly that: it says WHEN
    the write landed, so it can be lined up against what else was running.
    """
    target = real_settings_store() if path is None else path
    now = real_store_signature(target)
    if now != previous:
        try:
            when = f"{target.stat().st_mtime_ns / 1e9:.3f}"
        except OSError:  # pragma: no cover - the file went away between the two reads
            when = "unknown"
        pytest.fail(
            f"something wrote the owner's real settings store {target} while this test ran: "
            f"(exists,size,sha256) {previous} -> {now}, at epoch {when}. Two ways that "
            f"happens, and this guard cannot tell them apart: (1) THIS test wrote it — the "
            f"suite redirects QSettings to {_SETTINGS_ROOT}, so look for a QSettings "
            f"constructed with an explicit NativeFormat, or a setPath/setDefaultFormat "
            f"call that undoes tests/conftest.py; (2) ANOTHER PROCESS wrote it — the "
            f"shipped app, or a test that launches the packaged app, writes recentFiles "
            f"and lastDir on quit, and a run of one during this test looks exactly like "
            f"this. Check the timestamp above against what else was running.")
    return now


#: Refreshed after a failure is reported, so ONE offending test is named and the tests
#: after it are not all failed for somebody else's write.
_REAL_STORE_SIG = real_store_signature()


@pytest.fixture(autouse=True)
def _real_settings_store_is_never_written():
    """FAIL the test that touched the owner's real preferences file.

    The redirect above is what prevents the write; this is what makes a future defeat of
    it impossible to miss, and it names the test rather than leaving a changed md5 to be
    noticed weeks later. Function-scoped on purpose: a session-scoped check reports "the
    suite wrote it" and nothing more. One small read + hash per test, measured at 16 µs on
    the owner's 1.4 kB plist — and the read is the whole point: see
    :func:`real_store_signature` for why an mtime comparison made this guard report writes
    that never happened, and :func:`_locate_real_settings_store` for why it no longer asks
    Qt to re-locate the store on every check.

    IT CANNOT TELL WHOSE WRITE IT WAS. If the shipped app is launched while the suite runs,
    the test in flight is the one failed. That is the trade: the alternative is a guard
    that never fails, which is the situation this replaces. The baseline is refreshed after
    every report, so one write cannot fail the rest of the run.
    """
    global _REAL_STORE_SIG
    yield
    try:
        check_real_store_untouched(_REAL_STORE_SIG)
    finally:
        # Whatever the answer, the NEXT test is compared against what is on disk now —
        # one write must not fail every test that follows it.
        _REAL_STORE_SIG = real_store_signature()


from nanoscope.core.document import CalibratedChannel
from nanoscope.core.types import (
    ChannelInfo,
    ChannelKind,
    Provenance,
    RecordingMeta,
    SampleRange,  # noqa: F401  (handy for range-based tests)
)


# ================================================================================================
# qapp — one QApplication for the whole session, honouring the bootstrap.
# ================================================================================================

@pytest.fixture(scope="session")
def qapp():
    """A single QApplication, created via the app's own factory (offscreen).

    Also performs the two v2 registrations the ported GUI/analysis code relies on,
    guarded so this fixture works before those modules land (Phases 5-8).
    """
    from nanoscope.gui.app import create_app

    app = create_app([])

    # THE SUITE MUST NOT READ THE DEVELOPER'S OWN SAVED WINDOW LAYOUT.
    # `create_app` names the org/app so QSettings has somewhere to key on, and
    # `MainWindow._save_settings` writes `geometry` and `windowState` into it on every
    # close — so ANY real launch of the app, a screenshot harness, or a `--self-test` run
    # leaves a layout behind that the next `MainWindow()` in a test restores instead of
    # its defaults. `test_window_layout.py` and `test_shown_recording.py` each already
    # carry a private namespace for exactly this reason ("here 'first launch' really means
    # first launch"); the same hazard reaches every OTHER geometry- or focus-sensitive
    # GUI test, and nothing was protecting those.
    #
    # Measured, this session: three tests failed on the developer's machine and passed the
    # instant `geometry` and `windowState` were deleted from
    # `com.nanoscope-local.NanoScope.plist` —
    # `test_analysis_launcher.py::test_a_real_window_drives_the_ladder_and_never_clips_below_the_minimum`
    # (density read 'short' where 'minimal' was asserted, because the restored dock widths
    # left the central column wider than a 600 px window implies),
    # `test_measure_tab.py::test_dragging_the_inspector_narrow_still_sticks_with_the_measure_tab_raised`
    # and `test_p3_regressions.py::test_the_documented_curation_keys_are_reachable_on_the_table`.
    # The plist had been written by this session's own bundle and screenshot runs.
    #
    # One namespace for the whole session, so the suite is reproducible on a machine where
    # somebody has actually used the app. Per-module fixtures that set their own namespace
    # still nest inside this one and restore to it.
    app.setOrganizationName("NanoScopeTest")
    app.setApplicationName("NanoScopeSuite")
    QtCore.QSettings().clear()

    # Register analysis modules (Phase 5) and overlay renderers (Phase 6) once,
    # if they exist yet. Guarded so Phase 0-4 test runs don't fail on the import.
    try:
        from nanoscope.analysis import register_all as _register_analysis
        _register_analysis()
    except Exception:  # pragma: no cover - forward-compat guard
        pass
    try:
        from nanoscope.gui.overlays import register_all_renderers as _register_overlays
        _register_overlays()
    except Exception:  # pragma: no cover - forward-compat guard
        pass

    return app


# ================================================================================================
# NO MODAL MAY BLOCK THE RUN — one guard at the Qt seam, not a list of method names.
# ================================================================================================
#
# THE SHAPE OF THE DEFECT. Every neutraliser below this one names a METHOD of ours. That works
# for a modal the USER's CLICK asks for — a test that does not click never meets it — and it
# fails for a modal a RESULT opens, because a result arrives on a queued signal inside whatever
# ``processEvents()`` pump the test happens to be running. ``AnalysisPanel._show_sweep`` cost days
# that way; measured again this round on ``MainWindow._on_failed`` / ``_on_analysis_failed`` /
# ``_on_probed``:
#
#     detection raises, modal neutralised -> 1 failed, 31.5 s, exact traceback
#     detection raises, modal live        -> still running at 75 s, 0% CPU, no output, SIGKILLed
#
# and the hang is NOT curable by a more aggressive wait: the timeout timer does fire under the
# modal and does call ``quit()``; ``exec()`` simply cannot return while a nested event loop sits
# above it. So a failing analysis EATS THE REGRESSION THAT CAUSED IT and presents as "the suite
# hangs" instead of "test X failed".
#
# THE FIX IS THE CLASS, NOT THE THREE NAMES. There are 55 modal-opening call sites in
# ``nanoscope/gui/`` (AST-enumerated: every ``.exec()`` on a QDialog/QMessageBox/QMenu plus every
# static convenience box). Adding three names to a list leaves 52, and the next result-driven
# modal anyone writes reopens the hole. Instead we close the SEAM every one of them must pass
# through:
#
#   * ``QDialog.exec`` — the only ``exec`` in the whole dialog hierarchy's own ``__dict__``
#     (checked: QMessageBox, QFileDialog, QInputDialog, QColorDialog, QFontDialog and
#     QProgressDialog all inherit it), so ONE patch covers every dialog we have and every dialog
#     we will write.
#   * ``QMenu.exec`` — the other blocking ``exec`` (the File Manager's context menu). ASSIGNING
#     IT ON THE CLASS IS NOT ENOUGH, and that is all this guard used to do. Under
#     PySide6 6.11 ``QMenu.exec`` sits in the type's ``__dict__`` as a ``staticmethod`` (it has
#     a static overload), and Shiboken's instance ``tp_getattro`` answers ``menu.exec`` from
#     the C++ method table rather than from that dict — so ``QtWidgets.QMenu.exec`` read back
#     WAS the stub and ``is_modal_guard`` said True, while ``QMenu().exec`` was still
#     ``<built-in method exec>`` and opened a real nested loop. Measured: a probe that
#     right-clicked the File Manager sat there until a 500 s watchdog killed it, i.e. exactly
#     the "the suite hangs" failure this block exists to prevent, hiding behind a green
#     assertion that the guard was installed. ``QDialog.exec`` is an ordinary
#     ``method_descriptor`` and does not have this problem, which is why nobody noticed.
#     So the QMenu half is closed at the two seams that instances really do pass through:
#       - ``QMenu.__init__`` binds the stub ONTO EACH INSTANCE (that lookup wins over
#         Shiboken's, verified in ``tests/gui/test_s4_ig_modal_guard_reaches_qmenu.py``), which
#         covers every menu built from Python — including subclasses — before it can be shown;
#       - an application-wide event filter closes any QMenu that nevertheless becomes VISIBLE.
#         That is the net under menus Qt builds in C++ (``QMenuBar.addMenu``,
#         ``createStandardContextMenu``), whose ``__init__`` we never run, and under
#         ``popup()`` + a hand-rolled ``QEventLoop``, which is a blocking popup with no
#         ``exec`` in it at all.
#   * the STATIC convenience boxes. ``QMessageBox.warning`` and friends build and run their modal
#     in C++ and never touch ``QDialog.exec``; ``tests/gui/test_analysis_launcher.py`` already
#     documents having been bitten by exactly this. They are patched by name because Qt gives us
#     no seam under them.
#
# Every guard ANSWERS AS A CANCELLING USER WOULD — ``Rejected``, ``clickedButton() is None``, an
# empty path, ``ok=False``, an invalid colour — so a modal nobody meant to open destroys nothing
# and changes no number. It is recorded in ``blocked_modals`` so a test can assert what was said.
#
# WHAT IS DELIBERATELY *NOT* NEUTRALISED, and why:
#   * ``QApplication.exec`` (``gui/app.py``) — the app's own loop, not a modal; no test runs it.
#   * ``QEventLoop.exec`` — ``tests/gui/test_analysis_worker.py`` nests one ON PURPOSE.
#   * ``AnalysisPanel._show_comparison`` / ``EstimatorCompareDialog`` — the dialog's ANSWER is what
#     ``test_baseline_migration.py`` is about (it patches ``EstimatorCompareDialog.exec`` to return
#     Accepted and asserts the re-measure that branch launches). A previous attempt neutralised the
#     METHOD and silently deleted that branch. Patching the class-level seam is safe where
#     neutralising the method was not: the guard only supplies the DEFAULT answer (Rejected), and a
#     test's own patch of the concrete dialog class shadows it, so the Accepted branch is still
#     reachable and still tested.
#   * ``MainWindow._confirm_close_with_segments`` — its docstring claims it is injected "so the
#     suite never faces a live modal"; it never was. The guard's Cancel answer is the safe one
#     (nothing closed, no segment lost), so it is left to the guard rather than given a fourth
#     seam default; a test that wants a real answer shadows the method itself.
#
# The install is SESSION-SCOPED AND NEVER UNDONE. Restoring it between tests would only reopen the
# hole for the next one, and two existing tests restore ``QMessageBox.exec`` by hand from a plain
# ``getattr`` — which, once the guard is in place, re-installs the guard rather than the real
# ``exec``. That is the behaviour we want in a test process either way.

@dataclass(frozen=True)
class BlockedModal:
    """One modal that would have blocked the run, and what it was going to say."""
    kind: str          # concrete class, e.g. "QMessageBox", or "QMessageBox.warning"
    title: str
    text: str
    where: str         # "main_window.py:1242" — the line that opened it

    def __repr__(self) -> str:  # keeps assertion output readable
        return f"<{self.kind} @{self.where} {self.title!r}: {self.text!r}>"


_BLOCKED_MODALS: list[BlockedModal] = []
_GUARD_INSTALLED = False


def _record_modal(kind: str, title: object = "", text: object = "", *, depth: int = 2) -> None:
    import sys

    try:
        frame = sys._getframe(depth)
        where = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
    except Exception:  # pragma: no cover - defensive
        where = "?"
    _BLOCKED_MODALS.append(
        BlockedModal(kind=kind, title=str(title or ""), text=str(text or ""), where=where))


def _menu_offering(menu: object) -> str:
    """The entries a popup menu was going to show, as one readable line.

    Entries are joined with ``|``, a separator prints as ``—`` and a disabled entry is
    marked as such, so a test can assert on what the user would actually have been able to
    CHOOSE, not merely on what was drawn. Defensive throughout: this runs while a widget is
    mid-gesture.
    """
    try:
        parts = []
        for act in menu.actions():
            if act.isSeparator():
                parts.append("—")
                continue
            text = act.text()
            parts.append(text if act.isEnabled() else f"{text} (disabled)")
        return " | ".join(parts)
    except Exception:  # pragma: no cover - a dying C++ object
        return ""


#: How often the net below looks for a popup that is up. Small next to any hang and large
#: enough to cost nothing: a live popup stays up until something closes it, so the only
#: thing this interval sets is how long that takes.
_POPUP_POLL_MS = 25


class _PopupCloser(QtCore.QObject):
    """Close any QMenu that is actually up, and record what it offered.

    THE NET UNDER THE INSTANCE GUARD. ``_install_modal_guard`` binds a stub onto every menu
    built from Python, but a menu Qt builds in C++ (``QMenuBar.addMenu``,
    ``QLineEdit.createStandardContextMenu``) never runs that ``__init__``, and ``popup()``
    plus a hand-rolled ``QEventLoop`` is a blocking popup with no ``exec`` in it to patch.
    All of them end in the same observable state -- a VISIBLE popup with a nested event loop
    above the test -- so that is what is watched for, rather than any one call.

    A POLLING TIMER, NOT AN APPLICATION EVENT FILTER. The filter is the tidier spelling and
    it is not free: an app-level Python ``eventFilter`` is called for EVERY event delivered
    to every object, measured here at +1.9 us each (0.27 -> 2.15 us), which the whole GUI
    suite would pay on every mouse move and every repaint to catch a thing that happens a
    handful of times. A timer costs 40 wake-ups a second and works for the same reason
    conftest's header says a hang is not curable by waiting: the timer DOES fire under the
    modal -- it is ``exec`` that cannot return -- so the loop that is blocking the test is
    itself what runs the close.
    """

    def __init__(self) -> None:
        super().__init__()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(_POPUP_POLL_MS)
        self._timer.timeout.connect(self._sweep)
        self._timer.start()

    def _sweep(self) -> None:
        popup = QtWidgets.QApplication.activePopupWidget()
        if not isinstance(popup, QtWidgets.QMenu):
            return
        _record_modal("QMenu", popup.title(), _menu_offering(popup), depth=1)
        # close(), not hide(): a menu closed the way a user dismisses it lets its own
        # ``exec`` return (with no action chosen), which is the cancelling answer.
        popup.close()


#: The entry a test is "clicking" in the next popup menu, by visible text. Empty = the
#: default cancelling answer. Set through the ``menu_answer`` fixture, never by hand: the
#: fixture is what clears it again, and a leftover answer would silently drive somebody
#: else's menu.
_MENU_ANSWER: list[str] = []

_POPUP_CLOSER: _PopupCloser | None = None


def _ensure_popup_closer() -> bool:
    """Arm the visible-popup net, once a QApplication exists.

    Called from the guard install AND from the per-test ``blocked_modals`` fixture, because
    in some run orders the guard is installed before any QApplication exists and a QTimer
    needs one. Returns True when the net is armed.
    """
    global _POPUP_CLOSER
    app = QtWidgets.QApplication.instance()
    if app is None:
        return False
    if _POPUP_CLOSER is None:
        _POPUP_CLOSER = _PopupCloser()
    return True


def is_modal_guard(fn: object) -> bool:
    """True when ``fn`` is one of this guard's stubs rather than Qt's own blocking call.

    Tagged rather than sniffed by ``__module__``: a conftest can be imported under several
    names (pytest's ``conftest``, a subprocess's own alias) and Qt's C++ slots have no
    ``__module__`` at all.
    """
    return bool(getattr(fn, "_nanoscope_modal_guard", False))


def _install_modal_guard() -> None:
    """Replace every blocking modal entry point with a cancelling, recording stub."""
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return
    _GUARD_INSTALLED = True

    def _dialog_exec(self, *_a, **_k):
        title, text = "", ""
        try:
            title = self.windowTitle()
        except Exception:  # pragma: no cover - a dying C++ object
            pass
        try:
            if isinstance(self, QtWidgets.QMessageBox):
                text = " | ".join(p for p in (self.text(), self.informativeText()) if p)
        except Exception:  # pragma: no cover
            pass
        _record_modal(type(self).__name__, title, text)
        # Rejected, and no clicked button: every `box.exec(); box.clickedButton()` site in
        # nanoscope/gui falls to its Cancel/decline branch, which is the branch that keeps
        # the user's data.
        return QtWidgets.QDialog.DialogCode.Rejected

    def _menu_exec(self, *_a, **_k):
        # ...UNLESS A TEST SAID WHICH ENTRY IT IS CLICKING. Without this the guard makes the
        # context menu's own branches untestable: ``exec`` always answers None, so
        # ``chosen is act_only`` is never taken and no test can show that right-clicking a
        # recording and picking "Show only this recording" actually shows only it. The entry
        # is named by its VISIBLE TEXT (what a user picks) and it is ``trigger()``-ed, the
        # way Qt's own exec activates the action it returns, so a menu whose entries carry
        # slots behaves the same as under a real click.
        want = _MENU_ANSWER[-1] if _MENU_ANSWER else None
        if want is not None:
            for act in self.actions():
                if act.text() != want:
                    continue
                # A disabled entry is one a user CANNOT pick. Answering it anyway would let a
                # test "click" something the app had greyed out and call the result a
                # gesture, so it is an error, named as one.
                assert act.isEnabled(), (
                    f"menu_answer({want!r}) picked a DISABLED entry; the menu offered: "
                    f"{_menu_offering(self)}")
                _record_modal("QMenu", getattr(self, "title", lambda: "")(),
                              _menu_offering(self))
                act.trigger()
                return act
            raise AssertionError(
                f"menu_answer({want!r}) found no such entry; the menu offered: "
                f"{_menu_offering(self)}")
        # A popup menu the user never opened is a popup the user never chose from: None is
        # "dismissed without picking anything", the answer every ``chosen is act_x`` branch
        # in nanoscope/gui already handles. WHAT IT WAS GOING TO OFFER is recorded beside the
        # title, because for a context menu the title is empty and the entries are the whole
        # of what the user would have seen -- a test asserting the File Manager offers "Show
        # only this recording" should not have to open a real popup to find out.
        _record_modal("QMenu", getattr(self, "title", lambda: "")(), _menu_offering(self))
        return None

    _dialog_exec._nanoscope_modal_guard = True
    _menu_exec._nanoscope_modal_guard = True
    QtWidgets.QDialog.exec = _dialog_exec
    QtWidgets.QMenu.exec = _menu_exec
    # PySide6 keeps the PyQt-era ``exec_`` alias as a SEPARATE C++ entry in each class's own
    # ``__dict__``, so patching ``exec`` does not reach it. Nothing in nanoscope/gui spells it
    # that way today; a copy-pasted snippet from any Qt tutorial would.
    QtWidgets.QDialog.exec_ = _dialog_exec
    QtWidgets.QMenu.exec_ = _menu_exec

    # ...and the two class-level lines above are exactly the half-measure documented in the
    # header: they answer ``QtWidgets.QMenu.exec`` but not ``some_menu.exec``. The instance
    # lookup is closed HERE, at construction, which is the one moment every Python-built menu
    # (and every subclass of one) passes through before it can be shown.
    _qmenu_init = QtWidgets.QMenu.__init__

    def _guarded_qmenu_init(self, *a, **k):
        _qmenu_init(self, *a, **k)
        # A WEAKREF CLOSURE, NOT ``types.MethodType(_menu_exec, self)``. The bound method is
        # the obvious spelling and it SEGFAULTS the run: it puts a strong reference to the
        # menu's Python wrapper into that same wrapper's ``__dict__``, i.e. a reference cycle
        # on a widget whose C++ half Qt deletes on its own schedule, and the collector then
        # meets the wrapper after the object underneath it is gone. Measured on
        # ``tests/gui/test_adv_verify_gui.py``: 20 passed on three runs with the closure,
        # "Fatal Python error: Segmentation fault" mid-file on both runs with the bound
        # method. The tag below is what keeps ``is_modal_guard(menu.exec)`` answering True
        # either way.
        ref = weakref.ref(self)

        def _exec(*a2, **k2):
            menu = ref()
            return None if menu is None else _menu_exec(menu, *a2, **k2)

        _exec._nanoscope_modal_guard = True
        self.exec = _exec
        self.exec_ = _exec

    _guarded_qmenu_init._nanoscope_modal_guard = True
    QtWidgets.QMenu.__init__ = _guarded_qmenu_init
    _ensure_popup_closer()

    def _static(cls, name, answer, *, text_arg: int | None = 2):
        """Patch one C++ static convenience box to `answer` (a value or a callable)."""
        if not hasattr(cls, name):  # pragma: no cover - forward-compat with Qt versions
            return
        label = f"{cls.__name__}.{name}"

        def _stub(*args, **kwargs):
            title = args[1] if len(args) > 1 else kwargs.get("title", "")
            body = ""
            if text_arg is not None and len(args) > text_arg:
                body = args[text_arg]
            _record_modal(label, title, body)
            return answer() if callable(answer) else answer

        _stub._nanoscope_modal_guard = True
        setattr(cls, name, staticmethod(_stub))

    MB = QtWidgets.QMessageBox
    _static(MB, "information", MB.StandardButton.Ok)
    _static(MB, "warning", MB.StandardButton.Ok)
    _static(MB, "critical", MB.StandardButton.Ok)
    _static(MB, "about", None)
    _static(MB, "aboutQt", None, text_arg=None)
    # A question nobody answered is a question declined — never Yes.
    _static(MB, "question", MB.StandardButton.No)

    FD = QtWidgets.QFileDialog
    _static(FD, "getOpenFileName", ("", ""), text_arg=None)
    _static(FD, "getOpenFileNames", ([], ""), text_arg=None)
    _static(FD, "getSaveFileName", ("", ""), text_arg=None)
    _static(FD, "getExistingDirectory", "", text_arg=None)
    _static(FD, "getOpenFileUrl", (QtCore.QUrl(), ""), text_arg=None)
    _static(FD, "getOpenFileUrls", ([], ""), text_arg=None)
    _static(FD, "getSaveFileUrl", (QtCore.QUrl(), ""), text_arg=None)
    _static(FD, "getExistingDirectoryUrl", QtCore.QUrl(), text_arg=None)

    ID = QtWidgets.QInputDialog
    _static(ID, "getText", ("", False))
    _static(ID, "getMultiLineText", ("", False))
    _static(ID, "getInt", (0, False))
    _static(ID, "getDouble", (0.0, False))
    _static(ID, "getItem", ("", False))

    # An invalid QColor is precisely what a cancelled colour picker returns.
    _static(QtWidgets.QColorDialog, "getColor", QtGui.QColor, text_arg=None)
    _static(QtWidgets.QFontDialog, "getFont", lambda: (False, QtGui.QFont()), text_arg=None)


@pytest.fixture(scope="session", autouse=True)
def _modal_guard():
    """Install the guard once, before the first test, and leave it installed."""
    _install_modal_guard()
    yield


@pytest.fixture(autouse=True)
def blocked_modals(_modal_guard):
    """Every modal this test would have blocked on, in order.

    Autouse so the log cannot leak between tests; requestable by name so a test can assert
    that a failure REACHED the user — ``blocked_modals[0].text`` is what the box would have
    said. An empty list means the test opened no modal at all.
    """
    # The QApplication may not have existed when the guard was installed; the visible-popup
    # net needs one, and this is the fixture every test carries.
    _ensure_popup_closer()
    _BLOCKED_MODALS.clear()
    _MENU_ANSWER.clear()
    yield _BLOCKED_MODALS
    _BLOCKED_MODALS.clear()
    _MENU_ANSWER.clear()


@pytest.fixture
def menu_answer(_modal_guard):
    """Click one entry of the next popup menu, by the text the user would read.

    ``menu_answer("Show only this recording")`` makes the guard answer that menu the way a
    user picking that entry does — the action is triggered and returned, so the app's own
    ``chosen is act_x`` branch runs. Without an answer the guard cancels, which stays the
    default: a menu a test did not mean to open must still change nothing. Unknown or
    DISABLED entry ⇒ the test fails, naming what the menu actually offered.
    """
    def _answer(text: str | None) -> None:
        _MENU_ANSWER.clear()
        if text is not None:
            _MENU_ANSWER.append(str(text))

    yield _answer
    _MENU_ANSWER.clear()


# ================================================================================================
# Close-guard neutraliser — keep the unsaved-work modal from blocking teardown.
# ================================================================================================

@pytest.fixture(autouse=True)
def _no_close_guard_modal():
    """Default MainWindow's unsaved-work close prompt to a silent *Discard* for every test.

    pytest-qt closes each registered widget at teardown (``qtbot`` → ``w.close()``), which
    fires :meth:`MainWindow.closeEvent`; a window left with unsaved work + open documents
    would otherwise raise the real Save/Discard/Cancel modal and block the run forever. We
    override the injectable ``_confirm_close_choice`` seam at the class level so teardown
    always closes cleanly; the dedicated close-guard tests shadow it per-instance to assert
    the true decision table. Only patched when the GUI module is already imported, so pure
    analysis tests never pull in Qt on this account.
    """
    import sys

    mod = sys.modules.get("nanoscope.gui.main_window")
    if mod is None:
        yield
        return
    main_window = mod.MainWindow
    original = main_window._confirm_close_choice
    main_window._confirm_close_choice = lambda self: mod._CLOSE_DISCARD
    # The SAME hazard, for the second destructive confirmation: closing a recording that
    # carries detected events now asks before discarding them (a segment cannot be
    # reopened, so undo cannot reach past a close). Default it to "go ahead" so no test
    # that merely closes a detected document faces a live modal; the dedicated
    # close-guard tests shadow it per-instance to assert what it actually says.
    original_events = main_window._confirm_close_discarding_events
    main_window._confirm_close_discarding_events = lambda self, doc_ids: True
    # The THIRD, for the one destructive click undo cannot reach: deleting hand
    # measurements. Same pattern and same reason — default it to "go ahead" so an
    # existing test that removes rows keeps testing removal, not the dialog.
    mp_mod = sys.modules.get("nanoscope.gui.measure_panel")
    mp_original = None
    if mp_mod is not None:
        mp_original = mp_mod.MeasurePanel._confirm_remove
        mp_mod.MeasurePanel._confirm_remove = lambda self, n: True
    # The FOURTH is a DIFFERENT SHAPE and that is why it was missed. The three above are
    # confirmations the user's own CLICK asks for, so a test that does not click never
    # meets them. ``AnalysisPanel._show_sweep`` is a modal a RESULT opens: a finished
    # ``sweep_threshold`` raises SweepDialog on a queued signal, inside whatever
    # ``processEvents()`` pump happens to be running. Correct in the app; in a test it
    # means any test that lets that job finish hangs the whole run in ``QDialog.exec()``
    # with nothing to dismiss it — at 0% CPU, with no output, indistinguishable from a
    # slow test.
    #
    # Measured: ``test_run_gate.py::test_the_surrogate_chain_runs_off_the_gui_thread``
    # starts a real sweep to prove conditioning runs off the GUI thread, waits only until
    # that conditioning BEGINS, then pumps 20 more times before cancelling — so whether
    # the sweep's own result lands first is a race against machine speed, won on a loaded
    # machine and lost on an idle one. It hung deterministically in isolation here;
    # `sample` put the main thread in ``QDialog::exec()``. Fixed at the seam rather than in
    # that one test, because the next test to drive a sweep would recreate it.
    #
    # As of the seam guard above, NONE of the four below is load-bearing for the HANG any
    # more — ``QDialog.exec`` cannot block. They are kept because each supplies a DIFFERENT
    # default answer from the guard's decline: Discard, "yes, drop the events", "yes, remove
    # the rows", and a recorded sweep frame instead of a constructed SweepDialog. Deleting
    # them would silently flip those decisions, not merely re-expose a modal.
    #
    # NOT ``_show_comparison``, deliberately. That modal is reached only after a test has
    # itself called ``compare_baseline_estimators()``, and
    # ``test_baseline_migration.py::test_re_measure_now_inside_the_report_goes_through_the_same_entry_point``
    # patches ``EstimatorCompareDialog.exec`` to return Accepted and then asserts the
    # re-measure the Accepted branch launches. Neutralising the method here would silently
    # delete that branch and the test with it (measured: it fails on the fingerprint).
    # A per-test patch of the DIALOG is the right shape when the dialog's answer is what
    # the test is about; this fixture is for the modals nobody meant to open.
    #
    # The frame is recorded on the instance so a test can still assert what would have
    # been shown; SweepDialog's own behaviour is covered by tests that build it directly
    # (``test_panels.py``, ``test_p3_regressions.py``), which lose nothing here.
    ap_mod = sys.modules.get("nanoscope.gui.analysis_panel")
    ap_sweep = None
    if ap_mod is not None:
        panel_cls = ap_mod.AnalysisPanel
        ap_sweep = panel_cls._show_sweep

        def _record_sweep(self, sweep_df) -> None:
            self._last_sweep_df = sweep_df

        panel_cls._show_sweep = _record_sweep
    try:
        yield
    finally:
        main_window._confirm_close_choice = original
        main_window._confirm_close_discarding_events = original_events
        if mp_original is not None:
            mp_mod.MeasurePanel._confirm_remove = mp_original
        if ap_sweep is not None:
            ap_mod.AnalysisPanel._show_sweep = ap_sweep


# ================================================================================================
# Synthetic CalibratedChannel builder — the signal-first fixture the DSP tests consume.
# ================================================================================================

def make_synth_channel(
    data: np.ndarray,
    *,
    fs: float | None = 50_000.0,
    index: int = 0,
    name: str = "synthetic",
    units: str = "pA",
    kind: ChannelKind = ChannelKind.CURRENT,
    t0_s: float = 0.0,
) -> CalibratedChannel:
    """Build a target :class:`CalibratedChannel` around a raw numpy array.

    ``data`` is coerced to contiguous float32 (the CalibratedChannel contract).
    ``units`` populates both the channel units and, spelling-normalised, the
    ChannelInfo.raw_units. Pass ``fs=None`` to emulate the rate-unknown (legacy
    TDMS) case that gates ``requires_time_base`` modules.
    """
    arr = np.ascontiguousarray(np.asarray(data, dtype=np.float32))
    info = ChannelInfo(index=index, name=name, raw_units=units, kind=kind)
    return CalibratedChannel(
        info=info,
        data=arr,
        units=units,
        sample_rate_hz=fs,
        t0_s=t0_s,
    )


def make_recording_meta(
    *,
    n_samples: int,
    fs: float | None = 50_000.0,
    path: str = "synthetic.dat",
    channels: tuple[ChannelInfo, ...] | None = None,
    t0_s: float = 0.0,
) -> RecordingMeta:
    """A minimal RecordingMeta for context/module tests."""
    from pathlib import Path

    if channels is None:
        channels = (ChannelInfo(index=0, name="synthetic", raw_units="pA",
                                kind=ChannelKind.CURRENT),)
    rate_prov = Provenance.UNKNOWN if fs is None else Provenance.FILE_META
    return RecordingMeta(
        path=Path(path),
        format_name="synthetic",
        n_samples=n_samples,
        sample_rate_hz=fs,
        rate_provenance=rate_prov,
        channels=channels,
        t0_s=t0_s,
    )


def make_context(
    data: np.ndarray,
    *,
    fs: float | None = 50_000.0,
    params: dict | None = None,
    units: str = "pA",
    kind: ChannelKind = ChannelKind.CURRENT,
    path: str = "synthetic.dat",
):
    """Build a ``SimpleAnalysisContext`` around synthetic data.

    Guarded import: ``nanoscope.analysis.context`` lands in Phase 5. Until then
    this raises a clear skip-able error so conftest imports cleanly now.
    """
    try:
        from nanoscope.analysis.context import SimpleAnalysisContext
    except ImportError as exc:  # pragma: no cover - Phase 5 not built yet
        pytest.skip(f"SimpleAnalysisContext not available yet: {exc}")

    ch = make_synth_channel(data, fs=fs, units=units, kind=kind)
    meta = make_recording_meta(n_samples=ch.data.size, fs=fs, path=path,
                               channels=(ch.info,))
    return SimpleAnalysisContext(channel=ch, meta=meta, params=params or {})


# ================================================================================================
# live_doc_id — the ONE way a test crosses a session boundary.
# ================================================================================================

def live_doc_id(session, ref) -> str | None:
    """The session's CURRENT ``doc_id`` for the recording ``ref`` names, or None.

    ``ref`` may be a path or a ``doc_id`` from an EARLIER session — both are reduced to the
    source key, which is the identity that persists (``core.segment.persistent_key``).

    Why every project-restore test needs this. A ``doc_id`` names a LOAD, not a file
    (``nanoscope/core/identity.py``): reopening a recording is by construction a new load and
    gets a new id, so a test that carries a source window's ``doc_id`` into a restored
    session is asking about a load that no longer exists. That is not test friction — it is
    the guarantee under test. The session answering ``has_document`` for a previous load is
    precisely what let a detection that finished after its recording was closed and reopened
    install itself onto the wrong document, with all of its events past the end of the file.
    """
    from nanoscope.core.identity import source_of

    return session.doc_id_for_source(str(Path(source_of(str(ref))).resolve()))


# Expose builders as fixtures too, for tests that prefer injection.
@pytest.fixture
def synth_channel_factory():
    return make_synth_channel


@pytest.fixture
def context_factory():
    return make_context


# ================================================================================================
# Latched keyboard modifiers — a synthetic Ctrl+… outlives the test that sent it.
# ================================================================================================

@pytest.fixture(autouse=True)
def _no_latched_modifiers():
    """Leave ``QGuiApplication.keyboardModifiers()`` clear after every test.

    MEASURED: ``QTest.keyClick(w, Key_M, ControlModifier)`` leaves the application's
    modifier state at ``ControlModifier`` for the rest of the process — there is no
    matching release. ``QAbstractItemView.selectRow`` then asks ``selectionCommand``,
    which consults that state, and EXTENDS the selection instead of replacing it. Six
    tests in ``tests/gui/test_measure_navigation.py`` passed or failed according to
    whether ``test_p3_regressions.py`` (which sends a real Ctrl+M) had been collected
    first — i.e. the suite's greenness depended on alphabetical luck.

    Cleared here rather than in the one test that does it: any future test that sends a
    modified key has the same effect on every test after it, and the next one to do so
    should not have to rediscover this. The app's own ``MeasurePanel._select_source_row``
    was made modifier-independent separately — that is the user-facing half, since Ctrl+M
    IS the add shortcut and the modifier really is down when the row lands.
    """
    yield
    try:
        from PySide6 import QtCore, QtTest, QtWidgets
    except ImportError:                      # pragma: no cover - non-GUI runs
        return
    app = QtWidgets.QApplication.instance()
    if app is None or not app.keyboardModifiers():
        return
    receiver = app.activeWindow() or QtWidgets.QWidget()
    QtTest.QTest.keyClick(receiver, QtCore.Qt.Key_Escape, QtCore.Qt.NoModifier)
    app.processEvents()
