"""Run pytest so that one test that hangs or interrupts the run cannot hide the rest.

    python .github/scripts/run_tests_resumably.py --out-dir DIR [--idle-minutes 5]
        [--time-limit-minutes 70] pytest <pytest arguments...>

Everything after the word ``pytest`` goes to pytest. Why this exists: on GitHub's Windows
machines the test job twice ran into its time limit with no report. A step that is cut off
leaves nothing behind, and a process that inherited the step's output pipe can keep the step
"running" past its own time limit. So:

* pytest runs in its own hidden console (Windows) or session (elsewhere), with its output in
  files, so nothing it starts holds the step's pipe, and a Ctrl+C that a test sends to its
  console reaches only pytest and its children.
* nanoscope_ci_resume.py logs every test's start, result and finish as it happens.
* When pytest makes no progress for --idle-minutes, the driver kills its whole process tree.
  The test that was running is recorded as hung, with the stack dump that
  ``-o faulthandler_timeout=...`` wrote to stderr; when pytest ends inside a test (for example
  on a KeyboardInterrupt), that test is recorded too. Then pytest is started again and carries
  on after it.
* DIR/results.xml is a JUnit report of every test across the restarts, with each hung or
  interrupted test as an error. The exit code is 0 only when everything passed.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_RUNS = 40


def _usage() -> int:
    print(__doc__, file=sys.stderr)
    return 2


def _kill_tree(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(60)
    except subprocess.TimeoutExpired:
        pass


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _tail(path: Path, lines: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(could not read {path.name}: {exc})"
    return "\n".join(text.splitlines()[-lines:])


def _run_once(number: int, pytest_args: list[str], out_dir: Path, env: dict,
              idle_seconds: float, deadline: float) -> tuple[int | None, str | None]:
    """One pytest process. Returns (exit code, None), or (None, why it was killed)."""
    out = out_dir / f"run-{number:02d}.out.txt"
    err = out_dir / f"run-{number:02d}.err.txt"
    events = out_dir / "state" / "events.jsonl"
    command = [sys.executable, "-u", "-m", "pytest", "-p", "nanoscope_ci_resume", *pytest_args]
    print(f"== pytest run {number}: {' '.join(command)}", flush=True)
    options: dict = {}
    if sys.platform == "win32":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    else:
        options["start_new_session"] = True
    with open(out, "wb") as stdout, open(err, "wb") as stderr:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                env=env, **options)
        progress, last_progress = None, time.monotonic()
        while True:
            try:
                return proc.wait(5), None
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            seen = (_size(out), _size(events))
            if seen != progress:
                progress, last_progress = seen, now
            if now - last_progress > idle_seconds:
                _kill_tree(proc)
                return None, f"no progress for {idle_seconds / 60:g} minutes"
            if now > deadline:
                _kill_tree(proc)
                return None, "the overall time limit was reached"


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:  # a line cut short by a kill
            pass
    return events


def _testcase(suite: ET.Element, nodeid: str, seconds: float) -> ET.Element:
    path, _, name = nodeid.partition("::")
    case = ET.SubElement(suite, "testcase", classname=path, name=name or path)
    case.set("time", f"{seconds:.3f}")
    return case


def main(argv: list[str]) -> int:
    if "pytest" not in argv:
        return _usage()
    split = argv.index("pytest")
    options, pytest_args = argv[:split], argv[split + 1:]
    settings = {"--out-dir": None, "--idle-minutes": "5", "--time-limit-minutes": "70"}
    while options:
        key = options.pop(0)
        if key not in settings or not options:
            return _usage()
        settings[key] = options.pop(0)
    if not settings["--out-dir"]:
        return _usage()
    out_dir = Path(settings["--out-dir"]).resolve()
    state = out_dir / "state"
    state.mkdir(parents=True, exist_ok=True)
    for stale in ("events.jsonl", "skip.txt"):
        (state / stale).unlink(missing_ok=True)
    idle_seconds = float(settings["--idle-minutes"]) * 60
    deadline = time.monotonic() + float(settings["--time-limit-minutes"]) * 60

    env = dict(os.environ)
    env["NANOSCOPE_CI_STATE"] = str(state)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE), env.get("PYTHONPATH")]))

    stuck: list[tuple[str, str, str]] = []   # (nodeid, what happened, evidence)
    problem = None                           # something that went wrong outside any test
    done: set[str] = set()
    seen_events = 0
    for number in range(1, MAX_RUNS + 1):
        code, killed = _run_once(number, pytest_args, out_dir, env, idle_seconds, deadline)
        events = _read_events(state / "events.jsonl")
        new, seen_events = events[seen_events:], len(events)
        started = [e["nodeid"] for e in new if e["event"] == "start"]
        finished = {e["nodeid"] for e in new if e["event"] == "finish"}
        done.update(finished)
        open_tests = [nodeid for nodeid in started if nodeid not in finished]
        out_text = _tail(out_dir / f"run-{number:02d}.out.txt", 400)
        print(f"   run {number}: exit {code}, {len(finished)} finished, killed: {killed}", flush=True)
        if open_tests:
            if killed:
                what = f"hung: pytest was killed after {killed}"
            elif "KeyboardInterrupt" in out_text:
                what = "interrupted: pytest received a Ctrl+C (KeyboardInterrupt) during this test"
            else:
                what = f"pytest exited with code {code} during this test"
            evidence = (f"--- end of pytest's output (run {number}) ---\n"
                        + "\n".join(out_text.splitlines()[-60:])
                        + f"\n\n--- end of pytest's stderr (run {number}; stack dumps) ---\n"
                        + _tail(out_dir / f"run-{number:02d}.err.txt", 120))
            for nodeid in open_tests:
                stuck.append((nodeid, what, evidence))
            done.update(open_tests)
            (state / "skip.txt").write_text("\n".join(sorted(done)) + "\n", encoding="utf-8")
            if time.monotonic() > deadline:
                problem = "The overall time limit was reached; tests after the last one shown did not run."
                break
            continue
        if killed:
            problem = (f"pytest was killed outside any test after {killed}. The end of its output:\n"
                       + "\n".join(out_text.splitlines()[-60:]) + "\n\n--- stderr ---\n"
                       + _tail(out_dir / f"run-{number:02d}.err.txt", 120))
        elif code not in (0, 1, 5):  # 5: everything left was deselected
            problem = (f"pytest exited with code {code} outside any test. The end of its output:\n"
                       + "\n".join(out_text.splitlines()[-60:]) + "\n\n--- stderr ---\n"
                       + _tail(out_dir / f"run-{number:02d}.err.txt", 120))
        break
    else:
        problem = f"Stopped after {MAX_RUNS} restarts; later tests did not run."

    # One JUnit report across every run: hung/interrupted tests first, then everything else.
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", name="pytest")
    for nodeid, what, evidence in stuck:
        error = ET.SubElement(_testcase(suite, nodeid, 0.0), "error", message=what)
        error.text = evidence
    if problem:
        error = ET.SubElement(_testcase(suite, "pytest session", 0.0), "error",
                              message=problem.splitlines()[0])
        error.text = problem
    reports: dict[str, list[dict]] = {}
    for event in _read_events(state / "events.jsonl"):
        if event["event"] == "report":
            reports.setdefault(event["nodeid"], []).append(event)
    stuck_ids = {nodeid for nodeid, _, _ in stuck}
    counts = {"passed": 0, "failure": 0, "error": 0, "skipped": 0}
    for nodeid, parts in reports.items():
        if nodeid in stuck_ids:
            continue
        case = _testcase(suite, nodeid, sum(p.get("duration", 0.0) for p in parts))
        failed = [p for p in parts if p["outcome"] == "failed"]
        skipped = [p for p in parts if p["outcome"] == "skipped"]
        if failed:
            kind = "failure" if failed[0]["when"] == "call" else "error"
            node = ET.SubElement(case, kind, message=f"{failed[0]['when']} {kind}")
            node.text = "\n\n".join(p.get("text", "") for p in failed)
            counts[kind] += 1
        elif skipped:
            ET.SubElement(case, "skipped", message=skipped[0].get("text", "")[-300:])
            counts["skipped"] += 1
        else:
            counts["passed"] += 1
    ET.ElementTree(root).write(out_dir / "results.xml", encoding="utf-8", xml_declaration=True)

    print(f"== {counts['passed']} passed, {counts['failure']} failed, {counts['error']} errors, "
          f"{counts['skipped']} skipped, {len(stuck)} hung or interrupted"
          + (" - and the run did not finish" if problem else ""), flush=True)
    for nodeid, what, _ in stuck:
        print(f"   {what.split(':')[0].upper()}: {nodeid}", flush=True)
    if problem:
        print(problem, flush=True)
    return 0 if not (stuck or problem or counts["failure"] or counts["error"]) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
