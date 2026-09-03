"""H4 — the report window's header may not state a class count without its evidence.

The window used to compose its own sentence:

    f"{q.fit.k} subpopulation(s) at n_PCs={q.fit.n_pcs} "
    f"({100 * q.fit.conf_frac:.1f} % confidently assigned)"

which on the owner's -700 mV apoferritin rendered as *"7 subpopulation(s) at
n_PCs=11 (84.4 % confidently assigned)"* — over a partition scoring a 0.098 mean
silhouette, with a third of its events nearer another class's centre, on a cloud
where the independent density clusterer finds no cluster at all. Two of the
three surfaces that state the count are strings a reader carries away, and both
stated a count and a confidence with none of that beside it.

These tests pin the fix at its invariant rather than at its symptom: the window
does not get a nicer sentence, it stops writing the sentence. ``report.headline()``
is the single author of the claim, in the analysis layer beside the statistics
it has to be honest about, and this window repeats it verbatim — which is the
rule the window already followed for the caveats and did not follow for the
headline.

WHAT IS DELIBERATELY NOT TESTED HERE: the app's status bar. It is built in
``gui/main_window.py``, which this pass does not own; the string it prints is
still ``f"{q.fit.k} subpopulation(s) at n_PCs={q.fit.n_pcs}"``. That is reported
as a fence violation, not patched from inside this window.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. -----------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pathlib

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtCore

from nanoscope.analysis import subpop_report as SR
from nanoscope.analysis.events import REQUIRED, EventSet
from nanoscope.gui.subpop_window import SubpopReportWindow

FS = 50_000.0
N_SAMPLES = 400_000
N_EVENTS = 220


def test_nanoscope_resolved_from_one_real_tree_of_real_files():
    """Passes in BOTH trees, so a failure below cannot be an import that didn't happen."""
    root = pathlib.Path(nanoscope.__file__).resolve().parent
    from nanoscope.gui import subpop_window as SW
    for mod in (SR, SW):
        p = pathlib.Path(mod.__file__).resolve()
        assert root in p.parents
        assert p.stat().st_size > 20_000
    print(f"\n[H4] nanoscope resolved from {root}")


def _synthetic_events():
    rng = np.random.default_rng(0)
    x = np.full(N_SAMPLES, 200.0) + rng.normal(0.0, 2.0, N_SAMPLES)
    spans, pos = [], 500
    for i in range(N_EVENTS):
        deep = bool(i % 2)
        dwell = int(rng.integers(60, 140) if deep else rng.integers(15, 40))
        if pos + dwell + 400 >= N_SAMPLES:
            break
        x[pos:pos + dwell] -= 60.0 if deep else 30.0
        spans.append((pos, pos + dwell))
        pos += int(rng.integers(500, 1100))
    rows = []
    for i, (s, e) in enumerate(spans):
        depth = 60.0 if i % 2 else 30.0
        rows.append({
            "event_id": i, "recording": "r", "pore_id": "p1",
            "start_idx": s, "end_idx": e, "t_start_s": s / FS, "dwell_s": (e - s) / FS,
            "baseline_pA": 200.0, "depth_pA": depth, "frac_depth": depth / 200.0,
            "ecd_pA_s": depth * (e - s) / FS, "max_dev_pA": depth, "mean_dev_pA": depth,
            "rise_s": 1 / FS, "fall_s": 1 / FS, "snr": depth / 2.0, "n_levels": 1,
            "filter_limited": False, "accepted": True, "flag": "",
        })
    return (np.ascontiguousarray(x, dtype=np.float32),
            EventSet(pd.DataFrame(rows)[REQUIRED],
                     {"analyzed_duration_s": N_SAMPLES / FS, "polarity": -1}))


@pytest.fixture(scope="module")
def report():
    """A REAL report through the real pipeline — never a hand-written stub."""
    arr, events = _synthetic_events()
    cfg = SR.SubpopConfig(k_grid=(1, 2, 3), n_splits=2, n_boot=0)
    rep = SR.run_subpopulations(arr, arr, FS, events, recording="r",
                                duration_s=N_SAMPLES / FS, config=cfg)
    assert rep.quantification is not None
    return rep


@pytest.fixture(scope="module", autouse=True)
def _private_settings(qapp):
    org, app_name = qapp.organizationName(), qapp.applicationName()
    qapp.setOrganizationName("NanoScopeTest")
    qapp.setApplicationName("SubpopHonestyPrivate")
    QtCore.QSettings().clear()
    yield
    QtCore.QSettings().clear()
    qapp.setOrganizationName(org)
    qapp.setApplicationName(app_name)


@pytest.fixture
def window(qapp, qtbot, report):
    w = SubpopReportWindow()
    qtbot.addWidget(w)
    w.set_report(report, subtitle="synthetic")
    return w


def test_the_header_is_the_reports_headline_verbatim(window, report):
    """The window is not a second author of the claim.

    Not "the header mentions the silhouette" — the header IS the analysis
    layer's sentence, so a future change to how the claim is qualified reaches
    this surface without anyone remembering to update it.
    """
    head = window.header()
    assert report.headline() in head


def test_the_header_states_the_separation_beside_the_class_count(window, report):
    head = window.header()
    q = report.quantification
    assert f"{q.fit.k} subpopulation(s)" in head
    assert q.separation is not None
    assert f"{q.separation.silhouette:.3f}" in head
    assert q.separation.verdict in head
    # the count and the qualifier are one string, not two sentences a reader can split
    i_count = head.index(f"{q.fit.k} subpopulation(s)")
    i_sil = head.index(f"{q.separation.silhouette:.3f}")
    assert 0 < i_sil - i_count < 200


def test_the_header_no_longer_states_a_bare_count_and_confidence(window, report):
    """The exact string the -700 mV window printed must be unconstructable."""
    q = report.quantification
    old = (f"{q.fit.k} subpopulation(s) at n_PCs={q.fit.n_pcs} "
           f"({100 * q.fit.conf_frac:.1f} % confidently assigned)")
    assert old not in window.header()


def test_the_continuum_verdict_reaches_the_header_when_it_applies(window, report):
    verdict = report.continuum_verdict()
    if verdict:
        assert verdict in window.header()
        assert "ONE CONTINUUM" in window.header()


def test_the_caveat_count_is_still_on_the_header_and_still_points_at_the_strip(
        window, report):
    """The old reason for the old header is not lost in the rewrite."""
    head = window.header()
    n = len(report.warnings)
    assert f"{n} caveat" in head
    assert "beside the figures and at the bottom of the text" in head


def test_the_subtitle_still_leads_and_the_claim_is_on_its_own_line(window):
    head = window.header()
    assert head.startswith("synthetic\n")
    assert "detected region(s) passed the gate" in head.split("\n", 1)[1]


def test_an_empty_window_says_so_and_a_cleared_one_goes_back_to_it(window):
    window.set_report(None)
    assert "No subpopulation report" in window.header()


def test_the_caveat_strip_still_carries_every_warning_verbatim(window, report):
    """The header change must not have moved caveats OUT of the strip."""
    text = window.caveats()
    for w in report.warnings:
        assert w in text


def test_the_window_still_imports_no_analysis_module():
    """The header now says what the analysis layer says — by calling the report
    object it was handed, NOT by importing the analysis layer. The AST guard in
    ``test_subpop_views`` owns the general rule; this pins the specific risk this
    change introduced."""
    import ast
    from nanoscope.gui import subpop_window as SW

    tree = ast.parse(pathlib.Path(SW.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "analysis" not in node.module, node.module
        if isinstance(node, ast.Import):
            for a in node.names:
                assert "analysis" not in a.name, a.name
