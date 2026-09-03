"""B4DA-09 — the "did this test write the owner's preferences?" guard must not be the thing
that writes them, and must not report a write that did not happen.

``tests/conftest.py`` redirects QSettings so the suite can never key the owner's real store,
and an autouse fixture FAILS the test that touched it anyway. Two things were wrong with the
watching, both in the same direction — noise where the whole value is signal:

* it located the file by constructing ``QSettings(NativeFormat, UserScope, "nanoscope.local",
  "NanoScope")`` — on macOS that registers the domain with ``cfprefsd`` and destroying it
  runs ``CFPreferencesAppSynchronize`` on the owner's real domain — and it did that TWICE
  PER TEST, ~10 600 times in a full run;
* it compared ``(exists, size, mtime_ns)``, so a cfprefsd restamp of a byte-identical file
  read as "this test wrote the owner's real settings store". Measured: a ~20 s analysis test
  whose every assertion passed was reported as ``1 passed, 1 error``, the plist's mtime
  advancing ~20 s per run while its size never moved.

A guard that cries wolf is a guard that gets ignored, and this one is the evidence every
other fix in the suite rests on. The fix is one probe per PROCESS (taken at import, above the
baseline) and a CONTENT comparison. Neither half weakens it: hashing catches strictly more
than ``(size, mtime_ns)`` did, which is what the last two tests here are for.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6 import QtCore

import conftest as CT


@pytest.fixture
def stand_in(tmp_path):
    """A file to play the part of the real store. The guard is never proved by damaging the
    thing it guards — the same discipline ``test_rk_teardown_and_prefs_green2.py`` uses."""
    p = tmp_path / "com.example.StandIn.plist"
    p.write_bytes(b"<plist>original</plist>")
    return p


# ================================================================================================
# It no longer reports a write that never happened
# ================================================================================================
def test_a_restamped_mtime_is_not_a_write(stand_in):
    """THE DEFECT. cfprefsd restamps a registered domain when it flushes; the bytes are
    untouched, and untouched bytes are not a write."""
    before = CT.real_store_signature(stand_in)
    st = stand_in.stat()
    os.utime(stand_in, ns=(st.st_atime_ns, st.st_mtime_ns + 20_000_000_000))
    assert stand_in.stat().st_mtime_ns != st.st_mtime_ns, "the restamp did not take"

    assert CT.check_real_store_untouched(before, stand_in) == before, (
        "an mtime-only restamp was reported as 'this test wrote the owner's real settings "
        "store' — the false alarm that cost a green test an ERROR")


def test_the_signature_does_not_carry_the_mtime_at_all(stand_in):
    """Stated as the property, not as a behaviour: nothing downstream can start comparing
    timestamps again by accident."""
    sig = CT.real_store_signature(stand_in)
    exists, size, digest = sig
    assert exists is True
    assert size == len(stand_in.read_bytes())
    assert isinstance(digest, str) and len(digest) == 64, sig
    assert stand_in.stat().st_mtime_ns not in sig


def test_a_store_that_is_not_there_is_reported_as_absent(tmp_path):
    missing = tmp_path / "nothing.plist"
    assert CT.real_store_signature(missing) == (False, 0, "")
    # ...and appearing IS a change, which is what a first write looks like.
    before = CT.real_store_signature(missing)
    missing.write_bytes(b"x")
    with pytest.raises(pytest.fail.Exception, match="wrote the owner's real settings store"):
        CT.check_real_store_untouched(before, missing)


# ================================================================================================
# ...and it still fires — on strictly more than it used to
# ================================================================================================
def test_a_real_write_still_fails_the_test_that_made_it(stand_in):
    before = CT.real_store_signature(stand_in)
    stand_in.write_bytes(b"<plist>recentProjects: /tmp/pytest-of-nobody</plist>")
    with pytest.raises(pytest.fail.Exception) as excinfo:
        CT.check_real_store_untouched(before, stand_in)
    assert "wrote the owner's real settings store" in str(excinfo.value)
    assert stand_in.name in str(excinfo.value)


def test_an_in_place_edit_that_the_old_signature_would_have_missed_now_fires(stand_in):
    """THE STRICTNESS GAIN. Same length, same mtime — ``(exists, size, mtime_ns)`` could not
    see this, and a preferences write that swaps one path for another of equal length is
    exactly that shape."""
    before = CT.real_store_signature(stand_in)
    st = stand_in.stat()
    edited = bytearray(stand_in.read_bytes())
    edited[-2] = ord("X")
    stand_in.write_bytes(bytes(edited))
    os.utime(stand_in, ns=(st.st_atime_ns, st.st_mtime_ns))

    after = stand_in.stat()
    assert (after.st_size, after.st_mtime_ns) == (st.st_size, st.st_mtime_ns), (
        "the setup did not reproduce the invisible-to-the-old-guard case")
    with pytest.raises(pytest.fail.Exception, match="wrote the owner's real settings store"):
        CT.check_real_store_untouched(before, stand_in)


# ================================================================================================
# The guard does not touch the store it guards
# ================================================================================================
def test_checking_never_constructs_a_qsettings(monkeypatch, stand_in):
    """THE OTHER HALF OF THE DEFECT. Constructing a NativeFormat QSettings IS a touch on
    macOS; the guard did it twice per test purely to re-learn a path it already knew."""
    built: list[tuple] = []

    class _CountingSettings(QtCore.QSettings):
        def __init__(self, *a, **k):
            built.append(a)
            super().__init__(*a, **k)

    monkeypatch.setattr(QtCore, "QSettings", _CountingSettings)

    sig = CT.real_store_signature(stand_in)
    for _ in range(200):
        CT.real_settings_store()
        CT.check_real_store_untouched(sig, stand_in)

    assert built == [], (
        f"the guard constructed {len(built)} QSettings while merely CHECKING — every one of "
        f"them registers the owner's preferences domain with cfprefsd")


def test_the_store_is_located_once_and_the_answer_is_kept(monkeypatch):
    """The single probe happens at import, above the baseline snapshot, so any flush it
    provokes is part of the baseline instead of being blamed on whichever test is running."""
    def _boom():  # pragma: no cover - called only if the caching is gone
        raise AssertionError("the store was re-located after import")

    monkeypatch.setattr(CT, "_locate_real_settings_store", _boom)
    first = CT.real_settings_store()
    assert all(CT.real_settings_store() is first for _ in range(50))


def test_the_located_path_is_still_the_shipped_apps_own_store():
    """Cached, not hard-coded, and not the suite's private one. (Asserted WITHOUT building a
    probe of our own — a test that touches the domain to check the guard does not touch the
    domain would be the joke this file exists to end.)"""
    real = CT.real_settings_store()
    assert real.is_absolute(), real
    assert "NanoScope" in real.name, real
    assert CT._SETTINGS_ROOT not in real.parents, (
        f"the 'real store' the guard watches is inside the suite's private root: {real}")


def test_two_hundred_checks_leave_the_owners_real_store_byte_identical():
    """The guard, run against the real file the way every test runs it, changes nothing —
    read-only by construction (``open(..., 'rb')`` and a hash).

    If a foreign process (the shipped app, a packaging self-test) writes the plist during
    this test, that is a genuine write and the autouse guard reports it against this test —
    which is the documented trade, not a failure of this assertion.
    """
    sig = CT.real_store_signature()
    for _ in range(200):
        CT.real_store_signature()
    assert CT.real_store_signature() == sig


def test_the_guard_is_still_autouse_and_still_watches_the_real_store(request):
    assert "_real_settings_store_is_never_written" in request.fixturenames
