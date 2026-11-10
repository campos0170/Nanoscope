"""Print GitHub Actions ::error annotations for the failures in a pytest JUnit XML report.

Reading a job's log needs a signed-in GitHub account; a check run's annotations do not. Turning
test failures into annotations keeps a Windows run's results readable from the public API.

    python .github/scripts/junit_to_annotations.py report.xml "Windows tests"
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# GitHub keeps at most 10 error annotations per step: one summary plus this many details.
DETAILED = 9


def _message(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")


def _property(text: str) -> str:
    return _message(text).replace(":", "%3A").replace(",", "%2C")


def main(report: str, label: str) -> int:
    path = Path(report)
    if not path.exists():
        print(f"::error title={_property(label)}::"
              + _message(f"No test report at {report}: pytest did not get as far as writing one."))
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
    print(f"::error title={_property(f'{label}: {len(failures)} failed')}::{_message(listing[:30000])}")
    for name, _kind, body in failures[:DETAILED]:
        tail = "\n".join(body.splitlines()[-30:])
        print(f"::error title={_property(name[-200:])}::{_message(tail[:4000])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "Tests"))
