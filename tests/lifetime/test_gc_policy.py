"""Pin the deliberate exception to the suite-wide cyclic-GC policy."""

from __future__ import annotations

import gc


def test_lifetime_tests_explicitly_enable_aggressive_cyclic_gc():
    assert gc.isenabled(), "lifetime regressions must exercise Qt wrapper collection"
    assert gc.get_threshold() == (5, 1, 1)
