"""P15 (sweep-4 leftover) — a BROKEN preferences store is disclosed, not silently defaulted.

``_restorable_bytes`` already states the policy for one value ("*the preferences store is
external input, like any other file on disk* ... said out loud rather than swallowing"),
and ``_restore_settings`` disclosed wrong-TYPED values. What nothing asked about was the
FILE: a corrupted store (crash mid-quit, half-synced ``~/Library``) makes QSettings drop
keys wholesale with a ``FormatError`` — and the app came up factory-fresh (layout gone,
recent files gone) with nothing anywhere saying why, then quit and wrote a clean file over
the evidence.

The subtlety the fix has to carry: Qt parses a settings file ONCE per process and serves
every later ``QSettings`` from a cache that has forgotten the parse error (measured: first
instance ``FormatError``, second ``NoError``, same damaged file). So
:func:`main_window._settings_store_damaged` re-parses a COPY of the file at a fresh path —
which cannot be served from the cache — whenever ``status()`` claims health.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6 import QtCore

from nanoscope.gui.main_window import MainWindow, _settings_store_damaged

#: A malformation Qt itself refuses (unclosed section header): the one shape of damage
#: the platform can attest to, and therefore the one the disclosure is gated on.
_BROKEN = "[General\nlastDir=/somewhere\n"


def _store_path() -> Path:
    s = QtCore.QSettings()
    s.setValue("lastDir", "/Users/someone/data")
    s.sync()
    return Path(s.fileName())


def _poison_cache() -> None:
    """Model a real launch: something touches QSettings before _restore_settings, so the
    parse error is already swallowed by Qt's cache when the window asks."""
    early = QtCore.QSettings()
    early.value("lastDir")
    del early


def test_the_detector_sees_through_qts_cache(qapp):
    ini = _store_path()
    ini.write_text(_BROKEN)
    _poison_cache()
    cached = QtCore.QSettings()
    assert cached.status() == QtCore.QSettings.NoError, (
        "precondition: the cache hides the error — otherwise this test tests nothing")
    assert _settings_store_damaged(cached), (
        "the copy-and-reparse must report the file's real health")
    ini.unlink(missing_ok=True)


def test_a_broken_store_is_disclosed_on_the_info_bar(qapp, qtbot):
    ini = _store_path()
    ini.write_text(_BROKEN)
    _poison_cache()
    w = MainWindow()
    qtbot.addWidget(w)
    text = w._info_bar._label.text()
    assert "saved-preferences file is damaged" in text, (
        f"a store that drops keys wholesale must say so: {text!r}")
    assert "the next quit writes a healthy file over it" in text, (
        "the disclosure must name the remedy the app itself performs")
    ini.unlink(missing_ok=True)


def test_a_healthy_store_stays_silent(qapp, qtbot):
    """CONTROL: a false alarm on every launch is how the real one gets ignored."""
    ini = _store_path()
    _poison_cache()
    w = MainWindow()
    qtbot.addWidget(w)
    assert "damaged" not in w._info_bar._label.text()
    ini.unlink(missing_ok=True)


def test_a_missing_store_is_a_first_launch_not_damage(qapp, qtbot, tmp_path):
    ini = _store_path()
    ini.unlink()
    QtCore.QSettings().sync()   # let Qt notice the file is gone
    w = MainWindow()
    qtbot.addWidget(w)
    assert "damaged" not in w._info_bar._label.text()


def test_a_mangled_string_value_is_refused_and_named(qapp, qtbot):
    """Damage does not always break the file's grammar: truncation-plus-scribble can hand
    back a ``lastDir`` that is a perfectly good str full of control bytes, with
    ``status() == NoError``. This app never writes a control character into these keys,
    so one coming back is a value this app did not write — refused and disclosed through
    the existing ignored-keys sentence, never silently adopted as the next Open dialog's
    folder."""
    ini = _store_path()
    s = QtCore.QSettings()
    s.setValue("lastDir", "/Users/somebody/heal\x00\xff[broken")
    s.sync()
    del s
    w = MainWindow()
    qtbot.addWidget(w)
    assert "\x00" not in w._last_dir, "the mangled path must not be adopted"
    assert "lastDir" in w._info_bar._label.text(), (
        "the refused key is named — a dropped preference is never dropped silently")
    ini.unlink(missing_ok=True)


def test_the_next_quit_writes_a_healthy_file(qapp, qtbot):
    """The remedy the disclosure promises, performed: after _save_settings the store
    parses clean again and holds the app's values."""
    ini = _store_path()
    ini.write_text(_BROKEN)
    _poison_cache()
    w = MainWindow()
    qtbot.addWidget(w)
    w._save_settings()
    fresh = QtCore.QSettings()
    assert not _settings_store_damaged(fresh), "the rewritten store must parse clean"
    ini.unlink(missing_ok=True)
