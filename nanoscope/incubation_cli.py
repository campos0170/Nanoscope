"""Run a saved incubation study through the same engine used by the desktop app.

    python -m nanoscope.incubation_cli --manifest study.json --output new-result
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import threading


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Saved NanoScope study recipe JSON")
    parser.add_argument("--output", required=True, type=Path, help="New study result directory")
    args = parser.parse_args(argv)

    from .analysis.cancel import OperationCancelled
    from .analysis.incubation_study import load_spec, run_study

    cancelled = threading.Event()
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: cancelled.set())

    def progress(fraction: float, message: str) -> None:
        print(json.dumps({"progress": fraction, "message": message}), flush=True)

    try:
        spec = load_spec(args.manifest)
        result = run_study(spec, output_dir=args.output, progress=progress,
                           cancel=cancelled.is_set)
    except OperationCancelled:
        print("Study cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:  # scientific/input errors become a failed CLI run
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGINT, previous)
    print(json.dumps({"status": "complete", "output_dir": str(result.output_dir),
                      "recordings": len(result.summary), "events": len(result.events)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
