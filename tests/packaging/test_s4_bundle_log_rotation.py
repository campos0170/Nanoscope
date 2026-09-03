"""BD4-04 — two NanoScope instances starting at once must not destroy the kept log.

``packaging/nanoscope_app._redirect_stdio_to_log`` promises, in its own comment, to "keep one
previous run around". It used to do that with an unlocked check-then-act pair, so two starts that
coincided both saw the same oversized log: the first renamed it to ``nanoscope.log.1`` and opened
a fresh one, and the second renamed THAT fresh log on top of the backup. The previous session's
diagnostics were gone (measured at 114/200 natural concurrent starts before the fix), and one of
the two live instances spent its whole session writing into a file no longer reachable as
``nanoscope.log`` — where nobody, following INSTALL.txt, would ever look.

These tests drive the launcher itself in real subprocesses with HOME pointed at a scratch tree,
which is the gesture that reaches the bug: a shell/script double start, Login Items racing a
startup script, or a relaunch overlapping a still-shutting-down copy. Nothing is monkeypatched and
no sleep is injected into the rotation window.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PACKAGING = Path(__file__).resolve().parents[2] / "packaging"
MARKER = "PREVIOUS-SESSION-MARKER"
OVERSIZED = MARKER + "\n" + "y" * 2_500_000  # comfortably over _MAX_LOG_BYTES (2 MiB)
BANNER = "===== NanoScope start"

# The child is a NanoScope start: it calls the REAL _redirect_stdio_to_log() and then writes one
# line, which lands wherever that call pointed fd 1. `hold` keeps the process (and therefore its
# "I am live" lock) alive, the way a running app does.
#
# `start_at` is a wall-clock instant both children wait for before the call. Nothing inside the
# launcher is patched, delayed or synchronised — this only makes "the user started two copies at
# the same moment" reliable instead of a coin flip: unsynchronised Popen calls happen to stagger
# the two starts often enough that the old, broken code passed one run in three.
_CHILD = """
import os, sys, time
sys.path.insert(0, {pkg!r})
import nanoscope_app
start_at = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
while time.time() < start_at:
    pass
nanoscope_app._redirect_stdio_to_log()
print("HELLO-FROM-%s" % os.getpid())
sys.stdout.flush()
time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 0.0)
"""


def _child_script(tmp_path: Path) -> Path:
    path = tmp_path / "start_nanoscope.py"
    path.write_text(_CHILD.format(pkg=str(PACKAGING)))
    return path


def _log_dir(home: Path) -> Path:
    """Where the launcher puts the log on this platform (mirrors nanoscope_app._log_dir)."""
    if sys.platform == "darwin":
        return home / "Library" / "Logs" / "NanoScope"
    if sys.platform == "win32":
        return home / "AppData" / "Local" / "NanoScope" / "Logs"
    return home / ".local" / "state" / "NanoScope"


def _seed(home: Path, content: str = OVERSIZED) -> Path:
    d = _log_dir(home)
    if d.exists():
        for p in d.iterdir():
            p.unlink()
    else:
        d.mkdir(parents=True)
    (d / "nanoscope.log").write_text(content)
    return d


def _launch(script: Path, home: Path, count: int, hold: float = 0.0,
            together: bool = False) -> None:
    env = _env(home)
    start_at = time.time() + 1.0 if together else 0.0
    procs = [subprocess.Popen([sys.executable, str(script), str(hold), repr(start_at)], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for _ in range(count)]
    for p in procs:
        p.wait(timeout=120)


def _env(home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)  # Windows spelling of the same thing
    env["LOCALAPPDATA"] = str(home / "AppData" / "Local")
    return env


def _survey(d: Path) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for p in sorted(d.iterdir()):
        text = p.read_text(errors="replace") if p.is_file() else ""
        out[p.name] = {"size": p.stat().st_size, "marker": MARKER in text,
                       "hellos": text.count("HELLO-FROM-"), "banners": text.count(BANNER)}
    return out


def test_one_start_rotates_and_keeps_the_previous_run(tmp_path):
    """The control: started alone, the launcher does exactly what its comment promises."""
    script = _child_script(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    d = _seed(home)

    _launch(script, home, 1)

    files = _survey(d)
    assert files["nanoscope.log.1"]["marker"] is True, files
    assert files["nanoscope.log.1"]["size"] == len(OVERSIZED), files
    assert files["nanoscope.log"]["hellos"] == 1, files


def test_concurrent_starts_never_destroy_the_previous_run(tmp_path):
    """The defect: before the lock this lost the backup in well over half of the trials."""
    script = _child_script(tmp_path)
    home = tmp_path / "home"
    home.mkdir()

    for trial in range(8):
        d = _seed(home)
        _launch(script, home, 2, together=True)
        files = _survey(d)

        kept = [name for name, f in files.items() if f["marker"]]
        assert kept == ["nanoscope.log.1"], f"trial {trial}: previous session lost — {files}"
        assert files["nanoscope.log.1"]["size"] == len(OVERSIZED), f"trial {trial}: {files}"
        # And the second half of the defect: neither live instance may be stranded in the backup.
        assert files["nanoscope.log"]["hellos"] == 2, f"trial {trial}: {files}"
        assert files["nanoscope.log"]["banners"] == 2, f"trial {trial}: {files}"
        assert files["nanoscope.log.1"]["hellos"] == 0, f"trial {trial}: {files}"


def test_a_live_instances_log_is_not_rotated_away_from_it(tmp_path):
    """Rotation is skipped — and says so — while another instance is writing to the log.

    Renaming a file a live process holds open does not free anything: on POSIX that process keeps
    writing to the same inode, now called nanoscope.log.1, so the session the user is trying to
    report on ends up in the backup file. The skip is recorded in the log rather than being a
    silent no-op.
    """
    script = _child_script(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    d = _seed(home, content="")  # nothing to rotate yet

    holder = subprocess.Popen([sys.executable, str(script), "10"], env=_env(home),
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 60
        while "HELLO-FROM-" not in (d / "nanoscope.log").read_text(errors="replace"):
            assert time.time() < deadline, "the first instance never started"
            time.sleep(0.05)

        # It has been running a while and its log has grown past the cap.
        with (d / "nanoscope.log").open("a") as fh:
            fh.write("z" * 2_500_000)

        _launch(script, home, 1)  # a second instance starts now
        files = _survey(d)
    finally:
        holder.kill()
        holder.wait(timeout=60)

    assert "nanoscope.log.1" not in files, f"a live instance's log was rotated away: {files}"
    assert files["nanoscope.log"]["hellos"] == 2, files
    text = (d / "nanoscope.log").read_text(errors="replace")
    assert "was NOT rotated" in text and "another NanoScope instance is using it" in text, text
