"""Print GitHub Actions annotations for a pytest run: its failures, or where it stopped.

Reading a job's log needs a signed-in GitHub account; a check run's annotations do not. Turning
test results into annotations keeps a Windows run's results readable from the public API.

    python .github/scripts/junit_to_annotations.py report.xml "Windows tests" [--log out.txt] [--err err.txt]

* With a report: one ::error listing every failure, then details for the first few.
* Without one (pytest was killed, for example by the step's time limit): the last lines of the
  log, which name the test that was running, and the end of stderr, where pytest's
  faulthandler_timeout writes every thread's stack when a test runs too long.
* Either way, the log's "slowest durations" table as a ::notice, so slow tests are visible.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# GitHub keeps at most 10 error annotations per step: one summary plus this many details.
DETAILED = 9


def _message(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def _property(text: str) -> str:
    return _message(text).replace(":", "%3A").replace(",", "%2C")


def _annotate(level: str, title: str, text: str, limit: int = 4000) -> None:
    print(f"::{level} title={_property(title)}::{_message(text[-limit:])}")


def _read(path: str | None) -> list[str]:
    if not path or not Path(path).exists():
        return []
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:  # still held open by a process the time limit did not stop
        return [f"(could not read {path}: {exc})"]


def _durations(log: list[str]) -> str:
    """The "slowest N durations" table pytest prints with --durations, or ""."""
    for i, line in enumerate(log):
        if "slowest" in line and "durations" in line:
            table = [line]
            for row in log[i + 1:]:
                if not row.strip() or row.startswith("="):
                    break
                table.append(row)
            return "\n".join(table)
    return ""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("label", nargs="?", default="Tests")
    parser.add_argument("--log", help="pytest's stdout, written as it ran")
    parser.add_argument("--err", help="pytest's stderr, where faulthandler dumps go")
    args = parser.parse_args(argv)
    label = args.label
    log, err = _read(args.log), _read(args.err)

    slowest = _durations(log)
    if slowest:
        _annotate("notice", f"{label}: slowest tests", slowest)

    path = Path(args.report)
    if not path.exists():
        _annotate("error", label,
                  f"No test report at {args.report}: pytest did not finish. The end of its output:\n"
                  + ("\n".join(log[-60:]) or "(no output)"), limit=8000)
        if any(line.strip() for line in err):
            _annotate("error", f"{label}: stderr (stack dumps of a slow test)",
                      "\n".join(err[-120:]), limit=12000)
        return 0

    failures = []
    for case in ET.parse(path).getroot().iter("testcase"):
        for kind in ("failure", "error"):
            node = case.find(kind)
            if node is not None:
                name = f"{case.get('classname', '')}::{case.get('name', '')}"
                body = f"{node.get('message') or ''}\n{node.text or ''}".strip()
                failures.append((name, kind, body))
    if not failures:
        return 0
    listing = "\n".join(f"{kind.upper()}  {name}" for name, kind, _ in failures)
    _annotate("error", f"{label}: {len(failures)} failed", listing, limit=30000)
    for name, _kind, body in failures[:DETAILED]:
        _annotate("error", name[-200:], "\n".join(body.splitlines()[-30:]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
