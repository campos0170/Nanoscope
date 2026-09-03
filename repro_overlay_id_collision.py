"""MINIMAL REPRO — TraceView overlay identity is a CPython memory address.

nanoscope/gui/trace_view.py:531
    overlay_id = f"{doc_id}#{channel}:{hint.kind.value}:{id(hint)}"

``hint`` is a temporary that the caller drops as soon as add_overlay returns, and
CPython hands the very next OverlayHint the same address ~100% of the time. Two
different overlays of the same kind on the same (doc, channel) therefore get the
SAME id whenever no other allocation intervenes -- which is exactly what makes
tests/gui/test_measure_navigation.py::test_markers_are_redrawn_after_a_trace_view_rebuild
("redrawn, not a stranded dead reference") fail in ~1 of 3 `pytest tests/gui` runs.

    /opt/anaconda3/bin/python3 repro_overlay_id_collision.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from nanoscope.core.analysis import OverlayHint, OverlayKind


def make_hint(a: int) -> OverlayHint:
    return OverlayHint(OverlayKind.SPAN,
                       starts=np.array([a], np.int64),
                       stops=np.array([a + 10], np.int64),
                       color="#ff0000", label="x")


def overlay_id(doc_id: str, channel: int, hint: OverlayHint) -> str:
    """Verbatim from TraceView.add_overlay."""
    return f"{doc_id}#{channel}:{hint.kind.value}:{id(hint)}"


N = 2000
reuse = 0
prev = None
for i in range(N):
    h = make_hint(i)
    oid = overlay_id("/data/pore.abf", 0, h)
    del h                       # the caller's temporary goes away, as in _refresh_markers
    if prev is not None and oid == prev:
        reuse += 1
    prev = oid

print(f"consecutive overlays that received the SAME overlay_id: {reuse}/{N - 1} "
      f"({100 * reuse / (N - 1):.1f}%)")
print("overlay ids are NOT unique by construction; they are addresses.")
sys.exit(1 if reuse else 0)
