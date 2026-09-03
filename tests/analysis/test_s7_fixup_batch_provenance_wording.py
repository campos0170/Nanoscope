"""E4 (sweep-7 fix-up) -- a ``'batch'`` stamp says only what it knows.

`baseline.resolve_param_source` stamps ``'batch'`` on every recording the batch form
was NOT filled in for, and it does so for EVERY non-default form source alike -- a
confirmed proposal, a hand edit, a restored analysis -- without keeping which. The
first sweep-7 advisory (`batch_copied_warning`) nevertheless read "they were
proposed or confirmed for a DIFFERENT recording", which was measured false on the
owner's own files the first time a hand-typed form (1.5 s / 0.5 s, "set by hand")
was batch-detected: the -1000 mV result asserted a proposal and a confirmation that
never happened. The advisory is now built from `BATCH_PROVENANCE_CLAUSE`, which
names the recording the form was filled in on and the fact that this one was never
measured, and nothing about who chose the windows there.

These tests fail against the tree as it stood before the fix-up (the advisory
carried "proposed or confirmed") and cannot run against the pre-sweep-7 snapshot at
all (it has no ``'batch'`` source: wave 1 introduced it for S7-DET1).
"""
from __future__ import annotations

import numpy as np
import pytest

from nanoscope.analysis import baseline as B
from nanoscope.analysis.detect import detect_events

FS = 50_000.0

#: Provenance claims the stamp cannot back. Each names ONE of the form sources
#: `resolve_param_source` folds into ``'batch'``; a clause that carries any of them
#: is true for some batches and false for the others.
_NOT_KNOWN = ("proposed", "confirmed", "by hand", "restored")


def _trace(n=60_000, gap=2000, dwell=100, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.full(n, -100.0) + rng.normal(0.0, 2.0, n)
    t = 200
    while t + dwell < n:
        x[t:t + dwell] -= 60.0
        t += gap
    return x


def _claims(text: str) -> list[str]:
    low = text.lower()
    return [w for w in _NOT_KNOWN if w in low]


# ===========================================================================
# the advisory, for every form source the resolver folds into 'batch'
# ===========================================================================
@pytest.mark.parametrize("form_source", ["user", "proposed", "restored", "fallback", "batch"])
def test_the_batch_advisory_names_no_form_source_because_the_stamp_keeps_none(form_source):
    params = {"baseline_rough_s": 1.5, "baseline_base_s": 0.5,
              "baseline_param_source": form_source, B.BATCH_SCOPE_KEY: ("doc-A", 0)}
    cfg, src = B.baseline_config_from_params(params)
    assert src == form_source
    # Every one of these lands on the same stamp for the other recording...
    assert B.resolve_param_source(params, ("doc-B", 0), src) == B.BATCH_SOURCE
    # ...so the one sentence that explains the stamp must fit every one of them.
    adv = B.batch_copied_warning(cfg)
    assert _claims(adv) == [], adv
    # What IS known stays said: where the windows came from, their widths, and
    # that this recording was never probed.
    assert "COPIED FROM THE BATCH FORM" in adv
    assert "1.5 s / 0.5 s" in adv
    assert "DIFFERENT recording" in adv
    assert "never measured" in adv


def test_the_advisory_is_built_from_the_one_clause_the_panel_badge_can_share():
    clause = B.BATCH_PROVENANCE_CLAUSE
    assert "BATCH_PROVENANCE_CLAUSE" in B.__all__
    # Subject-less, so it reads after "...(1.5 s / 0.5 s): " and after
    # "These values: COPIED from the batch form: " alike.
    assert clause[0].islower(), clause
    assert _claims(clause) == [], clause
    assert "DIFFERENT recording" in clause and "never measured" in clause
    cfg = B.DEFAULT_BASELINE.replace(rough_s=1.5, base_s=0.5)
    assert clause in B.batch_copied_warning(cfg)


# ===========================================================================
# the seam and the provenance (what the .meta.json and the Stats box carry)
# ===========================================================================
def test_a_hand_typed_form_batch_detected_elsewhere_carries_no_proposal_claim_in_its_result():
    x = _trace()
    cfg = B.DEFAULT_BASELINE.replace(rough_s=0.2, base_s=0.05)
    res = B.characterisation_baseline(x, FS, config=cfg, polarity=-1,
                                      param_source=B.BATCH_SOURCE)
    hit = [w for w in res.warnings if "COPIED FROM THE BATCH FORM" in w]
    assert hit, res.warnings
    assert _claims(hit[0]) == [], hit[0]
    assert "0.2 s / 0.05 s" in hit[0] and "never measured" in hit[0]


def test_the_exported_baseline_warnings_of_a_batch_run_make_no_claim_the_stamp_cannot_back():
    df, prov = detect_events(_trace(), FS, polarity=-1, baseline_param_source="batch")
    assert prov["baseline_param_source"] == "batch"
    hit = [w for w in prov["baseline_warnings"] if "COPIED FROM THE BATCH FORM" in w]
    assert hit, prov["baseline_warnings"]
    assert _claims(hit[0]) == [], hit[0]
