"""Qt OBJECT-LIFETIME tests. The one suite that runs with the CYCLIC GC ENABLED.

WHY THIS DIRECTORY EXISTS
-------------------------
On 2026-08-31 a shipped build died with SIGSEGV in the owner's hands. He unticked a row in
the Plots window's "Legend entries" checklist; the slot redrew, the redraw rebuilt that very
list, and ``QListWidget.clear()`` destroyed the item Qt was still holding on its
``QStyledItemDelegate::editorEvent`` stack frame. The first click survived on freed-but-not-
yet-reused memory. The second one faulted.

THE SUITE COULD NOT HAVE CAUGHT IT, for two structural reasons, and this directory fixes
both:

1. THE REST OF THE SUITE RUNS WITH THE CYCLIC GC OFF. The in-tree root
   ``tests/conftest.py`` disables it before importing Qt, so the documented plain ``pytest``
   command cannot accidentally omit the policy. PySide6 6.11.1 deadlocks when the collector
   reclaims a Qt wrapper -- ``gc_collect_main`` -> ``SbkObject_tp_clear`` ->
   ``QObject::~QObject`` -> ``PyGILState_Ensure`` -> hang, sampled three times in one day.
   But the collector is exactly what RECYCLES the freed item between the user's two clicks,
   so switching it off hides this whole class of bug.
2. EVERY OTHER TEST DRIVES THE APP THROUGH ITS API. ``setCheckState()`` does not go through
   ``editorEvent``; only a real mouse release does. A lifetime bug that lives in Qt's own
   event machinery is invisible to a test that never enters it.

So: GC ON, real input events, and kept SMALL. The deadlock is a race that scales with the
number of collections, so a short suite is unlikely to hit it -- that is the trade, made
deliberately rather than by accident. If a run here ever hangs, it is the PySide6 deadlock and
not a product bug: sample the process (``sample <pid> 3``) and look for ``take_gil``.

RUN IT AS ITS OWN TARGET. This directory's fixture explicitly overrides the root policy:

    QT_QPA_PLATFORM=offscreen python -m pytest tests/lifetime -q
"""

from __future__ import annotations

import gc

import pytest

import nanoscope  # noqa: F401  -- Qt plugin-path bootstrap, as every GUI test does


@pytest.fixture(autouse=True)
def _cyclic_gc_on():
    """Enable the collector and make it AGGRESSIVE for the duration of each test.

    The default thresholds (700, 10, 10) collect far too rarely to reproduce a
    use-after-free inside a single event delivery. (5, 1, 1) forces a collection every few
    allocations, which is what turns "survives the first click" into a deterministic fault.
    Restored afterwards so this cannot leak into another test module.
    """
    was_enabled = gc.isenabled()
    old = gc.get_threshold()
    gc.enable()
    gc.set_threshold(5, 1, 1)
    try:
        yield
    finally:
        gc.set_threshold(*old)
        if not was_enabled:
            gc.disable()
        gc.collect()
