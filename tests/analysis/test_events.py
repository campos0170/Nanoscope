"""Tests for analysis.events: the EventSet contract and its 20-column SCHEMA.

These pin the cross-module schema (exact columns / order / dtypes), the
``empty_df`` dtypes, and the science guardrails on ``EventSet`` -- the
cross-pore pooling refusal and the ``rate_hz`` provenance requirement -- ported
from the SOURCE ``test_characterize.py`` schema assertions plus the EventSet
invariants in PORT_PLAN §3.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nanoscope.analysis.events import (
    REQUIRED,
    SCHEMA,
    EventSet,
    concat,
    empty_df,
)

# The frozen contract: exactly these 20 columns, in this order, with these
# dtypes. Reproduced verbatim from the SOURCE so a drift here fails loudly.
CANONICAL = [
    ("event_id", "int64"),
    ("recording", "string"),
    ("pore_id", "string"),
    ("start_idx", "int64"),
    ("end_idx", "int64"),
    ("t_start_s", "float64"),
    ("dwell_s", "float64"),
    ("baseline_pA", "float64"),
    ("depth_pA", "float64"),
    ("frac_depth", "float64"),
    ("ecd_pA_s", "float64"),
    ("max_dev_pA", "float64"),
    ("mean_dev_pA", "float64"),
    ("rise_s", "float64"),
    ("fall_s", "float64"),
    ("snr", "float64"),
    ("n_levels", "int64"),
    ("filter_limited", "bool"),
    ("accepted", "bool"),
    ("flag", "string"),
]


def _make_df(n: int, *, pore_id: str = "poreA", recording: str = "rec") -> pd.DataFrame:
    """A valid n-row event frame with the exact SCHEMA dtypes."""
    df = empty_df()
    for i in range(n):
        df.loc[i] = {
            "event_id": i,
            "recording": recording,
            "pore_id": pore_id,
            "start_idx": 100 * i,
            "end_idx": 100 * i + 50,
            "t_start_s": 0.001 * i,
            "dwell_s": 5e-4,
            "baseline_pA": -1000.0,
            "depth_pA": 100.0,
            "frac_depth": 0.1,
            "ecd_pA_s": 0.05,
            "max_dev_pA": 110.0,
            "mean_dev_pA": 100.0,
            "rise_s": 1e-5,
            "fall_s": 1e-5,
            "snr": 20.0,
            "n_levels": 1,
            "filter_limited": False,
            "accepted": True,
            "flag": "",
        }
    return df.astype({c: d for c, (d, _, _) in SCHEMA.items()})


# ================================================================================================
# SCHEMA shape / dtypes
# ================================================================================================

def test_schema_has_exactly_20_columns():
    assert len(SCHEMA) == 20
    assert len(CANONICAL) == 20


def test_schema_names_and_order_are_canonical():
    assert list(SCHEMA) == [name for name, _ in CANONICAL]
    assert REQUIRED == list(SCHEMA)


def test_schema_dtypes_match_canonical():
    for name, dtype in CANONICAL:
        assert SCHEMA[name][0] == dtype, name


def test_string_columns_are_the_pandas_extension_dtype_not_object():
    # PORT_PLAN §3.2: `string` is the pandas extension dtype, not object.
    for col in ("recording", "pore_id", "flag"):
        assert SCHEMA[col][0] == "string"


# ================================================================================================
# empty_df
# ================================================================================================

def test_empty_df_is_empty_with_full_schema():
    df = empty_df()
    assert len(df) == 0
    assert list(df.columns) == list(SCHEMA)


def test_empty_df_dtypes_match_schema():
    df = empty_df()
    for col, (dtype, _, _) in SCHEMA.items():
        assert df[col].dtype == pd.api.types.pandas_dtype(dtype), col


def test_empty_df_string_columns_are_extension_dtype():
    df = empty_df()
    for col in ("recording", "pore_id", "flag"):
        assert isinstance(df[col].dtype, pd.StringDtype), col


# ================================================================================================
# EventSet basics + validation
# ================================================================================================

def test_eventset_requires_all_columns():
    bad = empty_df().drop(columns=["depth_pA", "snr"])
    with pytest.raises(ValueError, match="missing required columns"):
        EventSet(bad)


def test_eventset_len_and_getitem():
    es = EventSet(_make_df(3))
    assert len(es) == 3
    assert list(es["event_id"]) == [0, 1, 2]


def test_accepted_property_filters_and_stamps_provenance():
    df = _make_df(3)
    df.loc[1, "accepted"] = False
    es = EventSet(df, {"recording": "rec"})
    acc = es.accepted
    assert len(acc) == 2
    assert acc.provenance["filtered"] == "accepted"


def test_n_pores_counts_distinct_pores():
    df = pd.concat([_make_df(2, pore_id="poreA"), _make_df(2, pore_id="poreB")], ignore_index=True)
    assert EventSet(df).n_pores == 2


# ================================================================================================
# rate_hz -- provenance requirement (invariant)
# ================================================================================================

def test_rate_hz_requires_analyzed_duration():
    es = EventSet(_make_df(4), {})
    with pytest.raises(ValueError, match="analyzed_duration_s missing"):
        es.rate_hz()


def test_rate_hz_uses_analyzed_duration():
    es = EventSet(_make_df(4), {"analyzed_duration_s": 2.0})
    assert es.rate_hz() == pytest.approx(2.0)


# ================================================================================================
# pool / concat -- cross-pore guardrail (invariant)
# ================================================================================================

def test_pool_same_pore_is_allowed():
    a = EventSet(_make_df(2, pore_id="poreA"), {"recording": "a"})
    b = EventSet(_make_df(3, pore_id="poreA"), {"recording": "b"})
    merged = a.pool(b)
    assert len(merged) == 5
    assert merged.provenance["pooled_from"] == ["a", "b"]


def test_pool_refuses_to_span_pores():
    a = EventSet(_make_df(2, pore_id="poreA"))
    b = EventSet(_make_df(2, pore_id="poreB"))
    with pytest.raises(ValueError, match="span"):
        a.pool(b)


def test_pool_spans_pores_with_explicit_optin():
    a = EventSet(_make_df(2, pore_id="poreA"))
    b = EventSet(_make_df(2, pore_id="poreB"))
    merged = a.pool(b, allow_multi_pore=True)
    assert len(merged) == 4
    assert merged.n_pores == 2


def test_concat_empty_returns_empty_eventset():
    es = concat([])
    assert len(es) == 0
    assert list(es.df.columns) == list(SCHEMA)


def test_concat_applies_same_cross_pore_guardrail():
    a = EventSet(_make_df(1, pore_id="poreA"))
    b = EventSet(_make_df(1, pore_id="poreB"))
    with pytest.raises(ValueError, match="span"):
        concat([a, b])
    ok = concat([a, b], allow_multi_pore=True)
    assert len(ok) == 2


# ================================================================================================
# repr smoke -- must not crash on empty or populated frames
# ================================================================================================

def test_repr_empty_and_populated():
    assert repr(EventSet(empty_df())) == "<EventSet empty>"
    r = repr(EventSet(_make_df(3)))
    assert "EventSet n=3" in r
