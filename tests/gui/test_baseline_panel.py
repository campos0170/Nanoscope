"""The propose-and-confirm baseline UI, and the guards that keep it a pure view.

WHAT THIS SURFACE IS FOR
========================
The characterisation baseline decides every ``depth_pA`` in the event table, and no
global default is correct for every recording: the owner's apo file needs a 2 s rough
window (a 500 ms window sitting inside its 521 ms clog *is* that state) and a gating
pore in the same corpus needs 30 ms. So the app measures and OFFERS -- and stops.

The stopping is the design, not politeness. Two measured facts say the app must not
decide alone: the usable noise estimator over-reads 3-4x on a gating pore, and the
reference's own QC triple is monotone in window length, so any search over it selects
the shortest window there is -- the exact defect this whole change removes.

THE UX RULES BEING PINNED
=========================
Each of these cost this codebase a round of rework once already:

* "locked" is STYLING, never ``setEnabled(False)`` -- Qt does not deliver a click to a
  disabled widget, so a disabled button can never answer "why not?";
* a gate reports the FIRST REAL blocker, not a downstream one;
* tooltips live on widgets;
* an un-scrolled ``QHBoxLayout`` of ``QPushButton`` s inside a dock floors that dock's
  width for good.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. ------------------------------
import os

import nanoscope  # noqa: F401,E402  -> runs bootstrap.py, fixes the plugin path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import ast  # noqa: E402
import pathlib  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

from nanoscope.analysis import baseline as B  # noqa: E402
from nanoscope.analysis import baseline_probe as P  # noqa: E402
from nanoscope.gui.baseline_panel import BaselinePanel  # noqa: E402

FS = 50_000.0


@pytest.fixture
def panel(qapp, qtbot):
    p = BaselinePanel()
    qtbot.addWidget(p)
    return p


def _proposal(rough=2.0, base=0.5, source="proposed"):
    probe = P.BaselineProbe(
        fs=FS, n=3_000_000, polarity=-1, sigma_pA=4.60, noise_pA=4.44,
        longest_state_s=0.496, n_states=489, mask_fraction=0.0931, gating=False,
        gating_score=0.51, level_dwell_s=None, probe_rough_s=5.0, usable=True,
        elapsed_s=0.23)
    cfg = B.DEFAULT_BASELINE.replace(rough_s=rough, base_s=base, decim=None)
    return P.BaselineProposal(
        config=cfg, param_source=source, probe=probe,
        reason="Longest state 496 ms (489 states, 9.3% of the trace) -> rough 2 s.")


# ===========================================================================
# 1. PROPOSE, NEVER DECIDE
# ===========================================================================
def test_a_proposal_changes_nothing_until_it_is_confirmed(panel):
    """THE central assertion of this surface. Showing a proposal must not move a
    single value -- if it did, the confirmation click would be decoration and the two
    measured reasons for asking (a sigma that over-reads on a gating pore, a QC
    objective that is monotone in window length) would be unanswered."""
    panel.set_recording(FS)
    before = panel.params()
    panel.set_proposal(_proposal())
    assert panel.params() == before
    assert panel.rough_s.value() == pytest.approx(B.DEFAULT_BASELINE.rough_s)

    assert panel.apply_proposal() is True
    assert panel.rough_s.value() == pytest.approx(2.0)
    assert panel.base_s.value() == pytest.approx(0.5)


def test_confirming_stamps_that_it_was_measured_and_not_defaulted(panel):
    """``baseline_param_source`` is the whole staleness story in one word, and no
    measurement can recover it afterwards: 'default' means nobody ever looked."""
    panel.set_recording(FS)
    assert panel.param_source() == "default"
    panel.set_proposal(_proposal())
    panel.apply_proposal()
    assert panel.param_source() == "proposed"
    _cfg, source = B.baseline_config_from_params(panel.params())
    assert source == "proposed"


def test_a_hand_edit_after_a_proposal_is_recorded_as_a_hand_edit(panel):
    panel.set_recording(FS)
    panel.set_proposal(_proposal())
    panel.apply_proposal()
    panel.rough_s.setValue(1.0)
    assert panel.param_source() == "user"
    assert B.baseline_config_from_params(panel.params())[1] == "user"


def test_the_proposal_is_rendered_with_the_measured_timescales_beside_it(panel):
    """3: "present the two measured timescales, the proposal, and a one-line reason"."""
    panel.set_recording(FS)
    panel.set_proposal(_proposal())
    assert panel.proposal_box.isVisible() or panel.proposal_box.isVisibleTo(panel)
    summary = panel.summary_label.text()
    for label, _value in P.proposal_summary(_proposal(), FS):
        assert label in summary
    assert "496 ms" in panel.reason_label.text()


def test_a_probe_that_could_not_decide_says_so_in_the_proposal_text(panel):
    """3: "If the probe cannot decide, fall back to 0.5/0.1/50 and SAY SO in the
    proposal text." A fallback that looked like a measurement would be worse than no
    proposal at all."""
    panel.set_recording(FS)
    fallback = P.propose_baseline(None)
    panel.set_proposal(fallback)
    text = panel.reason_label.text()
    assert "Could not measure" in text
    assert "0.5" in text and "0.1" in text
    panel.apply_proposal()
    assert panel.param_source() == "default"
    assert panel.rough_s.value() == pytest.approx(B.DEFAULT_BASELINE.rough_s)


def test_a_proposal_never_outlives_the_recording_it_was_measured_on(panel):
    """A proposal belongs to one file. Carried across a selection change it would be
    applied to another recording by a click that looks identical."""
    panel.set_recording(FS)
    panel.set_proposal(_proposal())
    assert panel.proposal() is not None
    panel.set_recording(100_000.0)
    assert panel.proposal() is None
    assert panel.apply_proposal() is False


# ===========================================================================
# 2. "LOCKED" IS STYLING
# ===========================================================================
def test_every_button_stays_clickable_while_it_is_locked(panel, qtbot):
    """Qt never delivers a click to a disabled widget, so a disabled button cannot
    explain itself. The whole point of a gate here is the explanation."""
    panel.set_gate(False, "Select a CURRENT channel to measure its baseline.")
    for b in (panel.probe_button, panel.apply_button, panel.reset_button):
        assert b.isEnabled(), f"{b.text()} was disabled; it must be dimmed, not disabled"


def test_clicking_a_locked_button_posts_the_reason_instead_of_swallowing_it(panel, qtbot):
    panel.set_gate(False, "Select a CURRENT channel to measure its baseline.")
    with qtbot.assertNotEmitted(panel.probeRequested):
        panel.probe_button.click()
    assert panel.gate_label.isVisibleTo(panel)
    assert "CURRENT channel" in panel.gate_label.text()


def test_the_gate_reports_the_first_real_blocker_and_not_a_downstream_one(panel):
    """With no recording selected, "press Measure first" would send the user to a
    button that is not reachable yet. The blocker they have to fix FIRST is the
    selection."""
    panel.set_gate(False, "Select a CURRENT channel to measure its baseline.")
    panel.apply_button.click()
    assert "CURRENT channel" in panel.gate_label.text()
    assert "Measure this recording" not in panel.gate_label.text()

    # ...and once that is fixed, the next real blocker appears in its place.
    panel.set_gate(True)
    panel.apply_button.click()
    assert "Measure this recording" in panel.gate_label.text()


def test_a_probe_in_flight_refuses_a_second_one_and_says_why(panel, qtbot):
    panel.set_gate(True)
    with qtbot.waitSignal(panel.probeRequested):
        panel.probe_button.click()
    assert panel.is_busy()
    with qtbot.assertNotEmitted(panel.probeRequested):
        panel.probe_button.click()
    assert "measuring" in panel.gate_label.text().lower()


def test_the_probe_button_says_it_is_working(panel):
    panel.set_busy(True)
    assert panel.probe_button.text() != "Measure this recording"
    panel.set_busy(False)
    assert panel.probe_button.text() == "Measure this recording"


# ===========================================================================
# 3. TOOLTIPS AND LAYOUT
# ===========================================================================
def test_every_control_carries_its_own_tooltip(panel):
    """Tooltips go on WIDGETS. (Never on a native macOS menu item, where they never
    render -- which is how this rule was learned.)"""
    for w in (panel.rough_s, panel.base_s, panel.decim_label, panel.probe_button,
              panel.apply_button, panel.reset_button, panel.source_label,
              panel.health_box, panel):
        assert w.toolTip().strip(), f"{w} has no tooltip"


def test_the_two_window_tooltips_state_the_constraint_that_sizes_them(panel):
    """A knob whose tooltip does not say what makes a value right or wrong is a knob
    the user cannot set. These two are the whole rule."""
    assert "longest" in panel.rough_s.toolTip().lower()
    assert "shorter" in panel.base_s.toolTip().lower()
    assert "521" in panel.rough_s.toolTip(), "cite the measurement, not an opinion"


def test_page_scrolling_bounds_the_form_without_obscuring_its_actions(qapp, qtbot):
    """The page now owns scrolling, not separate horizontal rows for each control.

    Keep the dock-width contract and verify all three action faces fit at a normal
    inspector width. Full-window dock resizing is covered in test_window_layout.
    """
    from nanoscope.gui.inspector_widgets import InspectorScrollArea

    panel = BaselinePanel()
    area = InspectorScrollArea(panel)
    qtbot.addWidget(area)
    area.resize(460, 700)
    area.show()
    QtWidgets.QApplication.processEvents()
    assert area.minimumSizeHint().width() < 300
    assert area.horizontalScrollBar().maximum() == 0
    for button in (panel.probe_button, panel.apply_button, panel.reset_button):
        area.ensureWidgetVisible(button)
        QtWidgets.QApplication.processEvents()
        assert button.visibleRegion().boundingRect().width() >= button.width() - 2
        assert button.visibleRegion().boundingRect().height() >= button.height() - 2


# ===========================================================================
# 4. THE HEALTH BLOCK
# ===========================================================================
def _health_dict(level="warn"):
    return {"checks": [
        {"name": "mask_fraction_floor", "level": level, "label": "mask fraction",
         "value": 0.024, "threshold": 0.025,
         "message": "the mask found no events (2.4% is the noise floor)"},
        {"name": "overshoot", "level": "ok", "label": "overshoot",
         "value": 0.09, "threshold": 0.3, "message": "0.09"},
    ]}


def test_the_health_checks_are_rendered_where_the_user_will_see_them(panel):
    """4/3: every one of the regimes these catch returns a finite, plausible, WRONG
    baseline with no exception anywhere. The checks are the only thing between the
    user and a silently mis-measured depth column."""
    panel.set_health(_health_dict())
    assert panel.health_box.isVisibleTo(panel)
    assert "mask fraction" in panel.health_label.text()
    assert "no events" in panel.health_label.text()


def test_a_restored_projects_health_block_renders_the_same_as_a_live_ones(panel):
    """A live run hands over a `BaselineHealth`; a reopened project hands over the
    JSON dict from provenance. If only one rendered, reopening a project would turn
    "this baseline is suspect" into silence."""
    live = B.BaselineHealth(checks=(
        B.BaselineCheck(name="mask_fraction_floor", level="warn",
                        label="mask fraction", value=0.024, threshold=0.025,
                        message="the mask found no events (2.4% is the noise floor)"),
    ))
    panel.set_health(live)
    from_live = panel.health_label.text()
    panel.set_health(None)
    panel.set_health({"checks": [c.to_dict() for c in live.checks]})
    assert panel.health_label.text() == from_live


def test_the_worst_check_is_shown_first(panel):
    panel.set_health({"checks": [
        {"level": "ok", "label": "fine", "message": "fine"},
        {"level": "refuse", "label": "fatal", "message": "fatal"},
        {"level": "warn", "label": "middling", "message": "middling"},
    ]})
    text = panel.health_label.text()
    assert text.index("fatal") < text.index("middling") < text.index("fine")


def test_a_clean_run_hides_the_health_block_rather_than_showing_an_empty_one(panel):
    panel.set_health(None)
    assert not panel.health_box.isVisibleTo(panel)


# ===========================================================================
# 5. PARAMS ROUND-TRIP
# ===========================================================================
def test_params_and_set_params_round_trip(panel):
    panel.set_recording(FS)
    panel.rough_s.setValue(0.03)
    panel.base_s.setValue(0.01)
    params = panel.params()

    other = BaselinePanel()
    other.set_recording(FS)
    other.set_params(params)
    assert other.params() == params
    cfg, _src = B.baseline_config_from_params(params)
    assert (cfg.rough_s, cfg.base_s) == (0.03, 0.01)


def test_params_carries_no_derived_value_that_could_go_stale(panel):
    """``to_provenance`` also emits ``baseline_fingerprint``, which is derived from
    the parameters AND from ``fs`` -- which this form does not resolve. A derived
    value sitting in a params dict is a value free to disagree with the parameters
    beside it, and it is not a detector keyword either (see
    :func:`test_a_full_provenance_block_in_params_never_reaches_the_detector`)."""
    assert set(panel.params()) == set(B.BASELINE_PARAM_KEYS) | {"baseline_param_source"}
    assert "baseline_fingerprint" not in panel.params()


def test_an_absent_key_leaves_its_widget_alone(panel):
    """A partial dict from an older provenance is a partial update -- never a reset to
    defaults over a table that was produced by other numbers.

    Seeded through `set_params` twice rather than by driving the spins, because a
    hand edit makes the form CONFIRMED and a confirmation is not seeded over (see
    `test_a_confirmed_form_is_never_seeded_over`)."""
    panel.set_params({"baseline_rough_s": 2.0, "baseline_base_s": 0.5,
                      "baseline_param_source": "restored"})
    assert panel.param_source() == "restored"
    panel.set_params({"baseline_rough_s": 1.0, "baseline_param_source": "restored"})
    assert panel.rough_s.value() == pytest.approx(1.0)
    assert panel.base_s.value() == pytest.approx(0.5)


def test_a_confirmed_form_is_never_seeded_over(panel):
    """THE REGRESSION. ``AnalysisPanel._seed_form_from`` runs on every selection, so
    one click on the channel that was ALREADY selected replaced the user's confirmed
    2.0 / 0.5 s with the stored table's 0.5 / 0.1 -- silently, and (because this
    method emits no ``paramsChanged``) without re-running the staleness check, so the
    banner that would have said the estimator no longer matches the table cleared
    too. §0.3 of this change is "the app proposes, the USER CONFIRMS"; a
    confirmation a navigation click can undo is not one."""
    panel.set_recording(FS, ("docA", 0))
    panel.rough_s.setValue(2.0)
    panel.base_s.setValue(0.5)
    assert panel.param_source() == "user"

    panel.set_params({"baseline_rough_s": 0.5, "baseline_base_s": 0.1,
                      "baseline_param_source": "restored"})
    assert (panel.rough_s.value(), panel.base_s.value()) == pytest.approx((2.0, 0.5))
    assert panel.param_source() == "user"

    # Re-pointing at the SAME target is navigation and costs nothing...
    panel.set_recording(FS, ("docA", 0))
    assert (panel.rough_s.value(), panel.base_s.value()) == pytest.approx((2.0, 0.5))
    assert panel.param_source() == "user"

    # ...but a different recording ends the confirmation, because the spec's first
    # sentence about these parameters is that they are per-recording.
    panel.set_recording(FS, ("docB", 0))
    assert panel.param_source() == "default"
    assert panel.rough_s.value() == pytest.approx(B.DEFAULT_BASELINE.rough_s)
    panel.set_params({"baseline_rough_s": 0.03, "baseline_base_s": 0.01,
                      "baseline_param_source": "restored"})
    assert (panel.rough_s.value(), panel.base_s.value()) == pytest.approx((0.03, 0.01))


def test_a_probe_proposal_the_user_applied_is_a_confirmation_too(panel):
    """``apply_proposal`` is THE confirmation click, and it stamps ``'proposed'``.
    Protecting only ``'user'`` would mean the one path the whole propose-and-confirm
    surface exists for is the one a selection change can undo."""
    panel.set_recording(FS, ("doc", 0))
    panel.set_proposal(_proposal(rough=2.0, base=0.5))
    assert panel.apply_proposal()
    assert panel.param_source() == "proposed"
    panel.set_params({"baseline_rough_s": 0.5, "baseline_base_s": 0.1,
                      "baseline_param_source": "restored"})
    assert (panel.rough_s.value(), panel.base_s.value()) == pytest.approx((2.0, 0.5))


def test_the_two_windows_can_never_be_put_in_an_order_the_estimator_refuses(panel):
    """REGRESSION. `BaselineConfig` refuses ``base_s > rough_s`` -- a short pass at
    least as wide as the rough one masks nothing and the two-pass structure is
    inoperative -- and the first version of this form let the user type straight into
    that state, where the ValueError surfaced inside a Qt slot with nowhere to go.

    So the two spins constrain each other, as this panel's sibling constrains
    ``k_out`` to just under ``k_in``. A partial update that would invert them clamps
    the other window rather than leaving the form describing something unrunnable.
    """
    panel.rough_s.setValue(2.0)
    panel.base_s.setValue(0.5)

    panel.set_params({"baseline_rough_s": 0.03})       # would invert the pair
    assert panel.base_s.value() <= panel.rough_s.value()
    panel.params()                                     # must not raise

    # ...and the coupling is ONE-directional, or the form is un-navigable: going
    # from 0.5 / 0.1 to a gating pore's 0.03 / 0.01 must not require lowering the
    # base window first, with the rough spin silently refusing to go down.
    panel.set_params({"baseline_rough_s": 2.0, "baseline_base_s": 0.5})
    panel.rough_s.setValue(0.03)
    assert panel.rough_s.value() == pytest.approx(0.03)

    # typing straight into the spin cannot reach it either
    panel.base_s.setValue(30.0)
    assert panel.base_s.value() <= panel.rough_s.value()
    panel.params()

    # ...and a legitimate widening still lands VERBATIM, both values, in one go --
    # applying them under the old bounds would have clamped the new rough window up.
    # Through a fresh panel: the spin edits above make this one CONFIRMED, and a
    # confirmation is deliberately not seeded over.
    fresh = BaselinePanel()
    fresh.set_params({"baseline_rough_s": 2.0, "baseline_base_s": 0.5})
    assert (fresh.rough_s.value(), fresh.base_s.value()) == pytest.approx((2.0, 0.5))


def test_back_to_default_returns_the_apps_one_literal(panel):
    panel.set_recording(FS)
    panel.rough_s.setValue(2.0)
    panel.reset_button.click()
    cfg, source = B.baseline_config_from_params(panel.params())
    assert cfg.fingerprint() == B.DEFAULT_BASELINE.fingerprint()
    assert source == "default"


def test_the_derived_decimation_is_shown_and_never_editable(panel):
    """2.4: cost goes as 1/decim^2, so it is the throttle, not a quality knob. The
    user sees it; they do not set it."""
    panel.set_recording(FS)
    panel.base_s.setValue(0.1)
    assert str(B.derive_decim(0.1, FS)) in panel.decim_label.text()
    assert not isinstance(panel.decim_label, QtWidgets.QAbstractSpinBox)
    assert panel.params()["baseline_decim"] is None


# ===========================================================================
# 6. THE REGRESSION THIS SURFACE CAUSED, FIXED AT THE INVARIANT
# ===========================================================================
def test_a_full_provenance_block_in_params_never_reaches_the_detector():
    """REGRESSION. The baseline form merges ``baseline_*`` keys into the detection
    params, and the first version merged the whole ``to_provenance`` block -- which
    also carries ``baseline_fingerprint``. ``_CONSUMED_PARAM_KEYS`` is built from
    `BASELINE_PARAM_KEYS`, the eight PARAMETER names, so the derived key sailed
    through and ``detect_events()`` raised ``unexpected keyword argument
    'baseline_fingerprint'`` on every single run. The window turned that into a modal
    error box, which hung the GUI suite.

    Fixed at the invariant rather than at the caller: the whole ``baseline_`` prefix
    belongs to `analysis.baseline` and none of it is a detector keyword, so the
    namespace is the boundary and the next derived key (``baseline_health``,
    ``baseline_median_pA``, ``baseline_fallback``) cannot reopen the gap.
    """
    from nanoscope.analysis.modules import _CONSUMED_PARAM_KEYS, _detect_kwargs

    block = B.DEFAULT_BASELINE.to_provenance(FS)
    assert "baseline_fingerprint" in block
    assert "baseline_fingerprint" not in _CONSUMED_PARAM_KEYS, (
        "the name list alone would have let it through -- that is the point")

    params = {"k_in": 5.0, **block, "baseline_param_source": "proposed",
              "baseline_health": {"checks": []}, "baseline_median_pA": -34.8}
    kwargs = _detect_kwargs(params)
    assert kwargs == {"k_in": 5.0}


def test_the_panel_merges_into_detection_params_without_colliding_with_them(panel):
    """The two forms share one params dict; a key owned by both would silently make
    one of them lose."""
    from nanoscope.gui.panels import DetectionPanel

    d = DetectionPanel()
    try:
        assert not (set(d.params()) & set(panel.params()))
    finally:
        d.deleteLater()


# ===========================================================================
# 7. STRUCTURAL GUARDS -- this widget is a pure view
# ===========================================================================
_MOD = pathlib.Path(__import__(
    "nanoscope.gui.baseline_panel", fromlist=["x"]).__file__)


def _tree():
    return ast.parse(_MOD.read_text())


def test_the_baseline_panel_imports_no_sibling_gui_panel():
    """MainWindow (and, for its own children, the owning AnalysisPanel) stays the sole
    wirer. Checked by parsing the imports rather than grepping, so a docstring that
    merely NAMES another panel -- this module's does -- is not mistaken for reaching
    it, and every spelling is caught."""
    reached: set[str] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("nanoscope.gui."):
                    reached.add(alias.name.split(".")[2])
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level == 1 and name:
                reached.add(name.split(".")[0])
            elif node.level == 1:
                reached |= {a.name for a in node.names}
            elif node.level == 2 and name.startswith("gui."):
                reached.add(name.split(".")[1])
            elif node.level == 0 and name.startswith("nanoscope.gui."):
                reached.add(name.split(".")[2])
    assert reached == set(), f"baseline_panel must import no sibling gui module: {reached}"
    # ...and the checker really would catch the import it stands guard over.
    guarded = ast.parse("from .analysis_panel import AnalysisPanel")
    assert any(isinstance(n, ast.ImportFrom) and n.level == 1
               and (n.module or "").split(".")[0] == "analysis_panel"
               for n in ast.walk(guarded))


def test_the_baseline_panel_measures_nothing_on_the_gui_thread():
    """A 3 M-sample probe is ~233 ms and an 8.6 M one ~579 ms. On the GUI thread that
    is a visible freeze, and a frozen window is a window the user clicks again. The
    owning AnalysisPanel runs it on the shared controller; this widget only renders
    the answer."""
    banned = {"probe_timescales", "probe_and_propose", "characterisation_baseline",
              "estimate_baseline", "estimate_baseline_result", "baseline_health",
              "preprocess_for_detection", "detect_events"}
    hits = []
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name in banned:
            hits.append(f"{name}@{node.lineno}")
    assert not hits, f"the baseline panel must not measure: {hits}"


def test_the_baseline_panel_constructs_no_baseline_config():
    """Same guard the analysis side carries: `DEFAULT_BASELINE` is the app's ONE
    baseline literal, and a widget that could spell its own would be a second one."""
    hits = [n.lineno for n in ast.walk(_tree())
            if isinstance(n, ast.Call)
            and (getattr(n.func, "id", None) == "BaselineConfig"
                 or getattr(n.func, "attr", None) == "BaselineConfig")]
    assert not hits, f"BaselineConfig constructed at {hits}"


def test_panels_module_still_hosts_the_baseline_page_without_wiring_it():
    """``panels.py`` may HOST the widget (as it hosts the measure tool) and must not
    build, read or connect it -- that module imports no sibling gui module and no
    analysis module beyond the kernels it already names."""
    import nanoscope.gui.panels as panels_mod

    src = pathlib.Path(panels_mod.__file__).read_text()
    tree = ast.parse(src)
    reached = {a.name.split(".")[2] for n in ast.walk(tree)
               if isinstance(n, ast.Import) for a in n.names
               if a.name.startswith("nanoscope.gui.")}
    reached |= {(n.module or "").split(".")[0] for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module}
    assert "baseline_panel" not in reached
    d = panels_mod.DetectionPanel()
    try:
        assert hasattr(d, "set_baseline_widget")
    finally:
        d.deleteLater()


def test_the_owner_wires_the_probe_and_merges_the_params():
    """The seam between the widget and the app: AnalysisPanel owns it (as it owns the
    measure and merge panels), merges its params into every run, and launches the
    probe off-thread."""
    import inspect

    from nanoscope.gui.analysis_panel import AnalysisPanel

    src = inspect.getsource(AnalysisPanel)
    assert "set_baseline_widget" in src
    assert "_on_probe_requested" in src
    assert "run_callable" in src
    # every run path goes through the one merging function
    assert src.count("self._baseline.params()") >= 2
    assert "_form_params" in src


# ===========================================================================
# 8. END TO END -- probe off-thread, confirm, run, and it is in the rows
# ===========================================================================
FS_E2E = 50_000.0
N_E2E = 400_000


def _doc(tmp_path):
    from nanoscope.core.document import TraceDocument
    from nanoscope.core.types import (
        ChannelInfo, ChannelKind, Provenance, RecordingMeta,
    )

    rng = np.random.default_rng(4)
    x = np.full(N_E2E, -100.0) + rng.normal(0.0, 5.0, N_E2E)
    for s in range(2000, N_E2E - 2000, 4000):
        x[s:s + 300] -= 60.0
    x = np.ascontiguousarray(x, dtype=np.float32)
    info = ChannelInfo(index=0, name="current", raw_units="pA",
                       kind=ChannelKind.CURRENT)
    meta = RecordingMeta(path=tmp_path / "pore.dat", format_name="synthetic",
                         n_samples=x.size, sample_rate_hz=FS_E2E,
                         rate_provenance=Provenance.FILE_META, channels=(info,))
    return TraceDocument(meta=meta, channels=(x,))


@pytest.fixture
def window(qapp, qtbot):
    from nanoscope.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w._analysis_controller.abort()


def test_the_probe_runs_off_the_gui_thread_and_lands_on_its_own_recording(
        window, qtbot, tmp_path):
    """4/5: the click emits intent; the OWNER measures, off-thread, on the SAME
    conditioned trace detection will use; the answer comes back as a proposal that
    still has to be confirmed."""
    session = window._session
    doc = _doc(tmp_path)
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)

    panel = window._analysis_panel
    bp = panel.baseline_panel
    assert bp.proposal() is None

    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        bp.probe_button.click()
    qtbot.wait(20)

    proposal = bp.proposal()
    assert proposal is not None and proposal.confident, proposal
    assert not bp.is_busy()
    # ...and it changed nothing until the click.
    assert bp.param_source() == "default"


def test_confirmed_windows_reach_the_rows_and_come_back_on_reselection(
        window, qtbot, tmp_path):
    """3: "values stay editable per recording and persist with the analysis".

    The full round trip: confirm a window, run detection, and the fingerprint stamped
    on the rows is the fingerprint of what the user confirmed -- then reselecting the
    recording puts those same windows back into the form, so the settings on screen
    are the settings that produced the table beside them.
    """
    session = window._session
    doc = _doc(tmp_path)
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)

    panel = window._analysis_panel
    bp = panel.baseline_panel
    bp.set_params({"baseline_rough_s": 2.0, "baseline_base_s": 0.5,
                   "baseline_param_source": "proposed"})

    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)

    res = panel.current_detection()
    prov = res.payload.provenance
    want = B.DEFAULT_BASELINE.replace(rough_s=2.0, base_s=0.5, decim=None)
    assert prov["baseline_fingerprint"] == want.fingerprint(FS_E2E)
    assert prov["baseline_rough_s"] == pytest.approx(2.0)
    assert prov["baseline_param_source"] == "proposed"

    # the form is re-seeded from the stored run on reselection
    bp.set_params({"baseline_rough_s": 0.5, "baseline_base_s": 0.1})
    window._on_selection(doc_id, 0)
    assert bp.rough_s.value() == pytest.approx(2.0)
    assert bp.base_s.value() == pytest.approx(0.5)


def test_the_health_block_of_the_run_on_screen_is_shown(window, qtbot, tmp_path):
    """3: "render the 2.7 health checks where the user will see them." Every regime
    they catch returns a finite, plausible, wrong baseline with no exception."""
    session = window._session
    doc = _doc(tmp_path)
    doc_id = session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)

    panel = window._analysis_panel
    with qtbot.waitSignal(window._analysis_controller.finished, timeout=30_000):
        panel.trigger_detection()
    qtbot.wait(20)

    health = panel.baseline_panel.health()
    assert health, "the detection's baseline health never reached the panel"
    assert panel.baseline_panel.health_label.text()


def test_a_probe_that_fails_or_is_cancelled_releases_the_button(window, qtbot,
                                                                tmp_path):
    """A busy widget that never comes back is indistinguishable from a hang -- and
    the panel refuses a second probe while it believes one is in flight, so a
    failure that left the flag set would lock the surface for the session."""
    doc = _doc(tmp_path)
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    panel = window._analysis_panel
    bp = panel.baseline_panel

    bp.set_busy(True)
    panel._on_controller_failed("boom")
    assert not bp.is_busy()

    bp.set_busy(True)
    panel.forget_pending_job()
    assert not bp.is_busy()


def test_the_baseline_gate_follows_the_detectors_gate(window, tmp_path):
    """A gate that could disagree with the detector's is a gate the user cannot
    reason about: both answer "can this recording be measured?"."""
    panel = window._analysis_panel
    bp = panel.baseline_panel
    assert bp._blocker("probe"), "no selection -> the first blocker is the selection"

    doc = _doc(tmp_path)
    doc_id = window._session.add_document(doc)
    window._create_default_layers(doc)
    window._on_selection(doc_id, 0)
    assert panel.can_run()
    assert not bp._blocker("probe")
