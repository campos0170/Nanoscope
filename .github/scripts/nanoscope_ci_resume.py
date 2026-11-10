"""pytest plugin used by run_tests_resumably.py; does nothing unless NANOSCOPE_CI_STATE is set.

It appends one JSON line per test event to $NANOSCOPE_CI_STATE/events.jsonl as the run goes
(start, each report, finish), so the driver knows exactly which tests finished, how, and which
one was running when pytest hung or was interrupted. It also deselects the node ids listed in
$NANOSCOPE_CI_STATE/skip.txt, which is how a restarted run carries on after that test.
"""
from __future__ import annotations

import json
import os

_STATE = os.environ.get("NANOSCOPE_CI_STATE")


def _append(record: dict) -> None:
    with open(os.path.join(_STATE, "events.jsonl"), "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def pytest_collection_modifyitems(session, config, items):
    if not _STATE:
        return
    path = os.path.join(_STATE, "skip.txt")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as stream:
        skip = set(stream.read().splitlines())
    dropped = [item for item in items if item.nodeid in skip]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = [item for item in items if item.nodeid not in skip]


def pytest_runtest_logstart(nodeid, location):
    if _STATE:
        _append({"event": "start", "nodeid": nodeid})


def pytest_runtest_logreport(report):
    if not _STATE:
        return
    record = {"event": "report", "nodeid": report.nodeid, "when": report.when,
              "outcome": report.outcome, "duration": report.duration}
    if report.failed:
        record["text"] = report.longreprtext[-8000:]
    elif report.skipped:
        record["text"] = report.longreprtext[-500:]
    _append(record)


def pytest_runtest_logfinish(nodeid, location):
    if _STATE:
        _append({"event": "finish", "nodeid": nodeid})
