"""Print GitHub Actions annotations for a JUnit test report.

Reading a job's log needs a signed-in GitHub account; a check run's annotations do not. Turning
test results into annotations keeps a Windows run's results readable from the public API.

    python .github/scripts/junit_to_annotations.py results.xml "Windows tests" [--log run.log]

GitHub keeps 10 annotations of each level per step and shows about 4 KB of each, so:

* each hung or interrupted test (run_tests_resumably.py lists those first) gets its own ::error
  with the end of pytest's output and the stack dump;
* every other failure is grouped by test file, one annotation per file, each test reduced to
  pytest's "file:line: Error" line and its first "E " lines. Files go out as ::error, then
  ::warning, then ::notice, so up to about 28 files are covered;
* one ::error first lists how many failed in each file, and a ::notice gives the slowest tests;
* with no report at all, one ::error says so, with the end of --log if given.
"""
from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

LIMIT = 3900  # characters GitHub shows of one annotation
PER_LEVEL = 10
SLOWEST = 25
_STUCK = ("hung", "interrupted", "pytest exited")
_LOCATION = re.compile(r"^\S.*:\d+: \w+")


def _message(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def _property(text: str) -> str:
    return _message(text).replace(":", "%3A").replace(",", "%2C")


def _annotate(level: str, title: str, text: str) -> None:
    print(f"::{level} title={_property(title[:250])}::{_message(text[:LIMIT])}")


def _tail(path: str | None, lines: int) -> str:
    if not path:
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(could not read {path}: {exc})"
    return "\n".join(text.splitlines()[-lines:])


def _short(body: str) -> str:
    """pytest's last "file:line: Error" line and the first few "E " lines of a failure."""
    lines = body.splitlines()
    where = [line for line in lines if _LOCATION.match(line)]
    errors = [line.strip()[:220] for line in lines if line.startswith("E ")][:4]
    if not errors:
        errors = [line.strip()[:220] for line in lines[:2] if line.strip()]
    return "\n".join(([f"  at {where[-1][:200]}"] if where else []) + [f"  {e}" for e in errors])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("label", nargs="?", default="Tests")
    parser.add_argument("--log", help="the test step's output, shown when there is no report")
    args = parser.parse_args(argv)
    label = args.label

    path = Path(args.report)
    if not path.exists():
        log = _tail(args.log, 60)
        _annotate("error", label, f"No test report at {args.report}."
                  + (f" The end of the test step's output:\n{log[-3000:]}" if log else ""))
        return 0

    cases = list(ET.parse(path).getroot().iter("testcase"))
    budget = {"error": PER_LEVEL, "warning": PER_LEVEL, "notice": PER_LEVEL}

    def emit(title: str, text: str, levels=("error", "warning", "notice")) -> bool:
        for level in levels:
            if budget[level]:
                budget[level] -= 1
                _annotate(level, title, text)
                return True
        return False

    timed = sorted(((float(c.get("time") or 0), c) for c in cases), key=lambda t: -t[0])
    if timed:
        rows = [f"{seconds:8.2f}s  {c.get('classname', '')}::{c.get('name', '')}"
                for seconds, c in timed[:SLOWEST]]
        emit(f"{label}: slowest of {len(cases)} tests", "\n".join(rows), levels=("notice",))

    stuck, by_file = [], {}
    for case in cases:
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is None:
                continue
            message = node.get("message") or ""
            body = node.text or ""
            nodeid = f"{case.get('classname', '')}::{case.get('name', '')}"
            if message.startswith(_STUCK):
                stuck.append((nodeid, message, body))
            else:
                by_file.setdefault(case.get("classname", ""), []).append(
                    (case.get("name", ""), kind, message, body))
    failed = len(stuck) + sum(len(v) for v in by_file.values())
    if not failed:
        return 0

    counts = [f"{len(v):4d}  {name}" for name, v in by_file.items()]
    counts = [f"{1:4d}  {nodeid}  ({message.split(':')[0]})" for nodeid, message, _ in stuck] + counts
    emit(f"{label}: {failed} failed in {len(by_file) + len(stuck)} places",
         "\n".join(counts) + "\n\nDetails follow, one annotation per file "
         "(errors, then warnings, then notices).")
    for nodeid, message, body in stuck:
        emit(nodeid, f"{message}\n{body[-(LIMIT - len(message) - 2):]}")
    for name, failures in by_file.items():
        parts, used = [], 0
        for index, (test, kind, message, body) in enumerate(failures):
            piece = f"- {test}  [{message or kind}]\n{_short(body)}"
            if used + len(piece) > LIMIT - 60:
                parts.append(f"... and {len(failures) - index} more in this file")
                break
            parts.append(piece)
            used += len(piece) + 1
        if not emit(f"{name}: {len(failures)} failed", "\n".join(parts)):
            break
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
