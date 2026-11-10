"""Print GitHub Actions annotations for a JUnit test report.

Reading a job's log needs a signed-in GitHub account; a check run's annotations do not. Turning
test results into annotations keeps a Windows run's results readable from the public API.

    python .github/scripts/junit_to_annotations.py results.xml "Windows tests" [--log run.log]

* A ::notice with the slowest tests (from each testcase's time).
* One ::error listing every failure and error, then details for the first few. The report from
  run_tests_resumably.py lists hung or interrupted tests first, so those get details.
* No report at all: one ::error saying so, with the end of --log if given.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# GitHub keeps at most 10 error annotations per step: one summary plus this many details.
DETAILED = 9
SLOWEST = 25


def _message(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def _property(text: str) -> str:
    return _message(text).replace(":", "%3A").replace(",", "%2C")


def _annotate(level: str, title: str, text: str) -> None:
    print(f"::{level} title={_property(title)}::{_message(text)}")


def _tail(path: str | None, lines: int) -> str:
    if not path:
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(could not read {path}: {exc})"
    return "\n".join(text.splitlines()[-lines:])


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
                  + (f" The end of the test step's output:\n{log}" if log else ""))
        return 0

    cases = list(ET.parse(path).getroot().iter("testcase"))
    timed = sorted(((float(c.get("time") or 0), c) for c in cases), key=lambda t: -t[0])
    if timed:
        rows = [f"{seconds:8.2f}s  {c.get('classname', '')}::{c.get('name', '')}"
                for seconds, c in timed[:SLOWEST]]
        _annotate("notice", f"{label}: slowest of {len(cases)} tests", "\n".join(rows))

    failures = []
    for case in cases:
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is not None:
                name = f"{case.get('classname', '')}::{case.get('name', '')}"
                body = f"{node.get('message') or ''}\n{node.text or ''}".strip()
                failures.append((name, kind, body))
    if not failures:
        return 0
    listing = "\n".join(f"{kind.upper()}  {name}" for name, kind, _ in failures)
    _annotate("error", f"{label}: {len(failures)} failed", listing[:30000])
    for name, _kind, body in failures[:DETAILED]:
        lines = body.splitlines()
        # Keep the first lines (what happened) and the last (where), within GitHub's limits.
        shown = lines if len(lines) <= 70 else lines[:10] + ["[...]"] + lines[-60:]
        _annotate("error", name[-200:], "\n".join(shown)[-12000:])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
