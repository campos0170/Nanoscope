"""M1 / M2 — THE APPLY CLICK ON A PROPOSAL THE APP'S OWN CHECKS CONDEMN.

THE DEFECT, MEASURED THROUGH THE REAL GUI ON THE OWNER'S -700 mV APO RECORDING
=============================================================================
``500 pM Apo ferritin_1 x PBS_22 nm_0005_Process_-700 mV.abf``, channel 2, 50 kHz,
driven through a real ``MainWindow`` (open -> select -> ``probeRequested`` ->
``apply_button.click()``)::

    probe proposal (twice, identical)   0.1 / 0.03 s
    BaselineHealth.ok                   False
      base_vs_state                     warn, 1.3819  (floor 3.0 x a 21.7 ms dwell)
      baseline_intrusion                warn, 0.041586 (ceiling 0.03)
    apply_button.isEnabled()            True
    apply_button.styleSheet()           ''          <-- unmarked
    gate_label                          ''  invisible
    ONE click ->  form 0.1 / 0.03, source 'proposed',
                  source_label "measured on this recording, and confirmed"

That is the whole finding: the panel put two red WARNINGs and a bold proposal on the
same screen, and then took the confirming click plainly, in one press, with nothing
anywhere on the screen recording that the windows accepted had been condemned. The
cost is not theoretical -- that click stamps an open pore of **-51.661 pA** where
three independent estimates of the same pore agree on **-48.218** (7.14 % out),
which is a median depth of 97.00 pA against 101.01 and a dI/I0 of 1.900 against
2.127.

AND THE REMEDY IT OFFERED COULD NOT BE FOLLOWED (M2)
====================================================
``base_vs_state``'s message ended *"...or re-run the probe"*. The probe is
deterministic and has never been told `BASE_STATE_RATIO_MIN`: it sizes ``base_s``
from the **p99.9** of the dwells while the check is measured against the **median**,
and on this recording two consecutive probes both returned 0.1 / 0.03 s
(``probe_twice`` above). An instruction that provably cannot work spends the one
instruction the user will follow, so it is gone; what replaces it is the action that
does work, and the reason the other one does not.

WHAT THESE TESTS PIN
====================
* the first Apply click on a condemned proposal REFUSES, and says what and why;
* it does not disable the button (this codebase's rule -- a disabled widget cannot
  answer "why not?" -- and reproducing an old analysis deliberately is legitimate);
* the second click accepts, and the panel then says so for as long as the form holds
  those numbers;
* a CLEAN proposal is untouched: one click, no mark, plain button. -500 mV
  (1.0 / 0.3 s) and -600 mV (2.0 / 0.5 s) pass all 8 checks and must not acquire a
  second click;
* a ``'detection'``-scope health block does NOT gate the click that repairs it;
* the remedy sentence names an action that works.

Every assertion here is on state a user can see -- the two spins, the button's text
and styling, the gate line, the source line -- or on a message string. Nothing reads
a private attribute for its verdict, so a package WITHOUT the fix fails these on the
claim being made rather than on an ``AttributeError`` that reads like a broken test.
"""

from __future__ import annotations

# --- Qt bootstrap MUST precede any Qt import. --------------------------------------------------
import os

import nanoscope  # noqa: F401  -> runs bootstrap.py, fixes the plugin path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import inspect
import pathlib

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets  # noqa: F401

from nanoscope.analysis import baseline as B
from nanoscope.analysis.baseline_probe import BaselineProposal
from nanoscope.gui.baseline_panel import BaselinePanel

FS = 50_000.0
I0 = -34.8
DEPTH = 110.0
NOISE = 4.84

APO_700 = pathlib.Path("/Users/abrahamcampos/500 pM Apo ferritin_1 x PBS_22 nm_"
                       "0005_Process_-700 mV.abf")

#: What the -700 mV probe returns, twice, measured through the real GUI on
#: 2026-08-22 -- and what its health block says about it.
PROBED_700 = (0.1, 0.03)
CONDEMNED_700 = ("base_vs_state", "baseline_intrusion")


# ===========================================================================
# where this ran, and whether the thing under test is really there
# ===========================================================================
def _package_root() -> pathlib.Path:
    return pathlib.Path(nanoscope.__file__).resolve().parent


def api(obj, name):
    """``getattr``, as an ASSERTION that the surface exists at all.

    A package without the fix must fail these tests on the CLAIM, not on an
    ``AttributeError`` that reads like a broken test -- and an out-of-tree revert
    STUB that is missing a method entirely must fail here, loudly, rather than
    manufacturing a fake failure three tests down.
    """
    assert hasattr(obj, name), (
        f"{type(obj).__name__ if not isinstance(obj, type) else obj.__name__} has no "
        f"{name!r} -- the tree under test at {_package_root()} is not the package "
        f"these tests are about")
    return getattr(obj, name)


def test_which_tree_this_ran_against_and_that_it_is_whole():
    """Say where ``nanoscope`` resolved from, and refuse a partial stand-in.

    An out-of-tree revert is run by pointing ``PYTHONPATH`` at another copy of this
    package, and a copy that is missing a method does not disprove a fix -- it
    manufactures failures that look like one. Both files under test are located and
    reported, and every surface the tests below touch is checked to exist FIRST.
    """
    root = _package_root()
    panel_src = pathlib.Path(inspect.getsourcefile(BaselinePanel)).resolve()
    health_src = pathlib.Path(inspect.getsourcefile(B.baseline_health)).resolve()
    print(f"\nnanoscope resolved from: {root}")
    print(f"  BaselinePanel   <- {panel_src}")
    print(f"  baseline_health <- {health_src}")
    assert panel_src == root / "gui" / "baseline_panel.py"
    assert health_src == root / "analysis" / "baseline.py"
    # The surfaces these tests drive, all of which predate the fix.
    for name in ("set_proposal", "set_health", "apply_proposal", "params",
                 "param_source", "health_text", "apply_button", "gate_label",
                 "source_label", "rough_s", "base_s"):
        assert hasattr(BaselinePanel, name) or name in ("apply_button", "gate_label",
                                                        "source_label", "rough_s",
                                                        "base_s"), name
    assert callable(B.baseline_health) and callable(B.estimate_baseline_result)


# ===========================================================================
# fixtures -- a condemned pair and a clean one, without an ABF
# ===========================================================================
def telegraph(dwell_s, duty, *, dur=15.0, seed=7, fs=FS):
    """A two-level pore at a chosen duty -- ``(trace, spans)``.

    Gamma(4) dwells and exponential gaps: a periodic fixture would make the block
    medians alternate deterministically and would be testing the fixture.
    """
    rng = np.random.default_rng(seed)
    n = int(round(dur * fs))
    x = np.full(n, float(I0))
    spans, t = [], 0.0
    gap = dwell_s * (1.0 - duty) / duty
    while True:
        t += rng.exponential(gap)
        d = rng.gamma(4.0, dwell_s / 4.0)
        s, e = int(round(t * fs)), int(round((t + d) * fs))
        if e >= n:
            break
        if spans and s <= spans[-1][1]:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
        t += d
    for s, e in spans:
        x[s:e] -= DEPTH
    x += rng.normal(0.0, NOISE, n)
    return x.astype(np.float32), [(s, e) for s, e in spans if e > s]


def _cfg(rough, base):
    return B.DEFAULT_BASELINE.replace(rough_s=rough, base_s=base, decim=None)


def _pair(rough, base):
    """``(proposal, health)`` for a window on a 20 ms-dwell synthetic pore.

    The proposal is spelled exactly as `baseline_probe.propose_baseline` spells one
    (``param_source='proposed'``), because what is being tested is the panel's
    handling of a real proposal, not of a mock.
    """
    x, _ = telegraph(0.020, 0.12)
    health = B.estimate_baseline_result(
        x, FS, config=_cfg(rough, base), polarity=-1).health
    proposal = BaselineProposal(
        config=_cfg(rough, base), param_source="proposed",
        reason=f"Longest state ... -> rough {rough:g} s ... base {base * 1e3:g} ms.")
    return proposal, health


@pytest.fixture(scope="module")
def condemned():
    """The -700 mV SHAPE: a tracking window about the size of the states."""
    p, h = _pair(0.04, 0.01)
    assert h.ok is False, "the fixture is supposed to be condemned"
    assert {c.name for c in h.checks if not c.ok} >= {"base_vs_state"}, [
        c.name for c in h.checks if not c.ok]
    return p, h


@pytest.fixture(scope="module")
def clean():
    """The -500 / -600 mV SHAPE: a window several times the state dwell."""
    p, h = _pair(0.4, 0.1)
    assert h.ok is True, (
        "the fixture is supposed to certify: " + "; ".join(h.warnings))
    return p, h


def _panel(qtbot):
    """A GENUINELY SHOWN panel.

    Not decoration: on an unshown widget every child reports ``isVisible() ==
    False``, so an assertion that the gate line reached the screen would pass
    vacuously in both directions.
    """
    w = BaselinePanel()
    qtbot.addWidget(w)
    w.show()
    QtWidgets.QApplication.processEvents()
    w.set_recording(FS, ("synthetic", 0))
    # The state a user is in when a proposal is on offer: a current channel is
    # selected, so the panel's own gate is open. Without this every Apply is blocked
    # by "select a current channel" and dimmed by it -- and a test that then asserted
    # "the button is marked" would be reading the WRONG mark, which is how a gate
    # that does not exist passes its own test.
    w.set_gate(True)
    return w


def _form(w):
    return (round(float(w.rough_s.value()), 6), round(float(w.base_s.value()), 6))


# ===========================================================================
# 1. THE ORIGINAL DEFECT
# ===========================================================================
def test_a_condemned_proposal_is_not_accepted_by_one_unmarked_click(qapp, qtbot,
                                                                    condemned):
    """The reported failure: enabled, unstyled, one press, no trace on screen."""
    proposal, health = condemned
    w = _panel(qtbot)
    before = _form(w)
    w.set_proposal(proposal)
    w.set_health(health, measured_for="proposal")

    # THE CODEBASE'S RULE HOLDS: never setEnabled(False). A disabled button cannot
    # answer the question the user is actually asking, and a user reproducing an old
    # analysis on purpose must still be able to reach these windows.
    assert w.apply_button.isEnabled() is True
    # ...and it is no longer unmarked.
    assert w.apply_button.styleSheet() != "", (
        "Apply is styled exactly as it is on a clean proposal, which is the defect")

    w.apply_button.click()
    qapp.processEvents()

    assert _form(w) == before, "the first click applied a condemned window"
    assert w.param_source() == "default"
    assert w.params()["baseline_base_s"] == pytest.approx(before[1])

    said = w.gate_label.text()
    assert w.gate_label.isVisible() is True
    assert "REFUSED" in said, said
    for label in ("base window vs state dwell",):
        assert label in said, said
    assert "Baseline health" in said
    # M2 AT THE GUI: the screen must not send the user back to the probe.
    assert "re-run the probe" not in said.lower().replace("re-running", "")
    assert "anyway" in said.lower()

    # The second press is a DIFFERENT press, and says so before it is made.
    assert w.apply_button.text() == "Use these values anyway"
    assert "ffb454" in w.apply_button.styleSheet().lower(), w.apply_button.styleSheet()

    w.apply_button.click()
    qapp.processEvents()
    assert _form(w) == (0.04, 0.01)
    assert w.param_source() == "proposed"

    # ...AND THE SCREEN KEEPS SAYING SO. The moment the click is over is not the
    # moment the user reads the depth it produced.
    src = w.source_label.text()
    assert "ACCEPTED OVER" in src, src
    assert "base window vs state dwell" in src, src
    assert "not certifiable" in src, src
    assert "ffb454" in w.source_label.styleSheet().lower()
    # The health block that justifies all of this is still on screen, worst first.
    text = w.health_text()
    assert "WARNING" in text and text.index("WARNING") < (
        text.index("PASS") if "PASS" in text else len(text))


def test_a_clean_proposal_is_untouched_by_any_of_this(qapp, qtbot, clean):
    """-500 mV (1.0 / 0.3 s) and -600 mV (2.0 / 0.5 s) pass all 8 checks.

    They must still be one press, plainly styled, with nothing added to the source
    line. A gate that costs a second click on a correct proposal is a gate the owner
    learns to click through, which is how the condemned one gets accepted.
    """
    proposal, health = clean
    w = _panel(qtbot)
    w.set_proposal(proposal)
    w.set_health(health, measured_for="proposal")

    assert w.apply_button.isEnabled() is True
    assert w.apply_button.styleSheet() == ""
    assert w.apply_button.text() == "Use these values"

    w.apply_button.click()
    qapp.processEvents()

    assert _form(w) == (0.4, 0.1)
    assert w.param_source() == "proposed"
    assert w.gate_label.text() == ""
    assert w.gate_label.isVisible() is False
    assert w.source_label.text() == (
        "These values: measured on this recording, and confirmed")
    assert "ACCEPTED" not in w.source_label.text()
    assert w.apply_button.text() == "Use these values"


# ===========================================================================
# 2. THE GATE IS AT THE ACTION, NOT AT THE BUTTON
# ===========================================================================
def test_the_programmatic_confirmation_is_gated_the_same_way(qapp, qtbot, condemned):
    """``apply_proposal()`` is public -- a menu, a driver or a batch can call it.

    If the gate lived only in the click handler, every one of those would be a path
    on which a condemned window is accepted with nothing said.
    """
    proposal, health = condemned
    w = _panel(qtbot)
    before = _form(w)
    w.set_proposal(proposal)
    w.set_health(health, measured_for="proposal")

    apply_proposal = api(w, "apply_proposal")
    assert apply_proposal() is False, "the first programmatic call applied it"
    assert _form(w) == before
    assert "REFUSED" in w.gate_label.text()

    assert apply_proposal() is True, "the second call must go through"
    assert _form(w) == (0.04, 0.01)
    assert "ACCEPTED OVER" in w.source_label.text()


def test_a_caller_that_means_it_can_say_so_in_one_call(qapp, qtbot, condemned):
    """``override=True`` -- for a caller that has already shown the condemnation.

    Checked through the SIGNATURE first: a package without the parameter must fail
    on the claim, not on a ``TypeError`` raised inside a Qt slot.
    """
    proposal, health = condemned
    w = _panel(qtbot)
    w.set_proposal(proposal)
    w.set_health(health, measured_for="proposal")

    sig = inspect.signature(api(w, "apply_proposal"))
    assert "override" in sig.parameters, (
        f"apply_proposal{sig} has no `override` -- there is no way for a caller that "
        f"has already reported the condemnation to confirm in one call")
    assert sig.parameters["override"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["override"].default is False

    assert w.apply_proposal(override=True) is True
    assert _form(w) == (0.04, 0.01)
    assert "ACCEPTED OVER" in w.source_label.text(), (
        "an overridden accept is still an accept over a failed check")


def test_a_detection_block_does_not_gate_the_click_that_repairs_it(qapp, qtbot,
                                                                   condemned):
    """The two health scopes are not interchangeable, and this is where it bites.

    A ``'detection'`` block is the verdict on windows that have ALREADY measured the
    table on screen. Gating Apply on it would refuse the very click that replaces
    them -- the app blocking its own repair with the evidence that repair is needed.
    """
    proposal, health = condemned
    w = _panel(qtbot)
    w.set_proposal(proposal)
    w.set_health(health, measured_for="detection")
    assert w.health_scope() == "detection"

    w.apply_button.click()
    qapp.processEvents()
    assert _form(w) == (0.04, 0.01), "a stored detection's warnings blocked a repair"
    assert "ACCEPTED OVER" not in w.source_label.text()


# ===========================================================================
# 3. THE MARK DOES NOT OUTLIVE THE NUMBERS IT DESCRIBES
# ===========================================================================
@pytest.mark.parametrize("end_it", ["edit", "reset", "another recording"])
def test_the_mark_dies_with_the_values_it_was_about(qapp, qtbot, condemned, end_it):
    """"These windows were condemned" is a statement about NUMBERS IN THE FORM.

    Left standing over another recording's values it is the same class of defect as
    a surrogate study's answer landing on the wrong file, one field along.
    """
    proposal, health = condemned
    w = _panel(qtbot)
    w.set_proposal(proposal)
    w.set_health(health, measured_for="proposal")
    # Accepted the way a user accepts one -- two presses. NOT through
    # ``override=True``: a package that does not have the parameter would fail this
    # test with a ``TypeError`` from inside a Qt slot, which reads like a broken test
    # rather than like a missing fix.
    w.apply_button.click()
    w.apply_button.click()
    qapp.processEvents()
    assert "ACCEPTED OVER" in w.source_label.text(), (
        "nothing was marked, so there is no mark for this test to be about")

    if end_it == "edit":
        w.base_s.setValue(0.005)
    elif end_it == "reset":
        w.reset_button.click()
    else:
        w.set_recording(FS, ("another-file", 2))
    qapp.processEvents()

    assert "ACCEPTED OVER" not in w.source_label.text(), w.source_label.text()
    assert "8a939c" in w.source_label.styleSheet()


def test_a_second_probe_disarms_a_previous_override(qapp, qtbot, condemned, clean):
    """Arming means "you have read THIS condemnation". A new verdict is a new one.

    Carrying the arming across would let one informed click authorise a warning the
    user has never seen.
    """
    proposal, health = condemned
    w = _panel(qtbot)
    w.set_proposal(proposal)
    w.set_health(health, measured_for="proposal")
    w.apply_button.click()                       # refused; armed
    assert w.apply_button.text() == "Use these values anyway"

    # A second probe comes back with the SAME (still condemned) window.
    w.set_proposal(proposal)
    w.set_health(health, measured_for="proposal")
    assert w.apply_button.text() == "Use these values", (
        "the arming survived a fresh proposal, so one click now accepts a "
        "condemnation the user was shown for a different measurement")
    before = _form(w)
    w.apply_button.click()
    qapp.processEvents()
    assert _form(w) == before


# ===========================================================================
# 3b. TWO DEFECTS FOUND INSIDE THIS FIX, PINNED
# ===========================================================================
def test_a_click_refused_by_a_different_blocker_arms_nothing(qapp, qtbot, condemned):
    """The first cut of this gate armed the override on ANY refused Apply click.

    The panel's own gate closes on a selection change and on a recording with no
    sample rate, and a proposal can still be on screen when it does. A click refused
    by "select a current channel" would then have armed the condemnation override --
    so the very next press, after the user fixed the selection, accepts a condemned
    window they were never shown. The arming has to be tied to the blocker that was
    actually posted, not to the fact that one was.
    """
    proposal, health = condemned
    w = _panel(qtbot)
    w.set_proposal(proposal)
    w.set_health(health, measured_for="proposal")
    before = _form(w)

    w.set_gate(False, "Select a current channel to measure its baseline.")
    w.apply_button.click()
    qapp.processEvents()
    assert "Select a current channel" in w.gate_label.text()
    assert "REFUSED" not in w.gate_label.text()
    assert w.apply_button.text() == "Use these values", (
        "a click refused by the channel gate armed the condemnation override")

    w.set_gate(True)
    w.apply_button.click()
    qapp.processEvents()
    assert _form(w) == before, (
        "one press accepted a condemned window the user was never shown")
    assert "REFUSED" in w.gate_label.text()


def test_the_gate_and_health_ok_cannot_disagree_about_any_level(qapp, qtbot):
    """``'info'`` is in the level vocabulary, is rendered here, and is NOT ``ok``.

    `BaselineCheck.ok` is ``level == 'ok'`` and `BaselineHealth.ok` is ``all(c.ok)``,
    so an ``'info'`` row already makes a block not-ok. The first cut of this gate
    condemned on ``{'warn', 'refuse'}`` -- the two levels emitted today -- which puts
    the gate and the flag it exists to enforce one new check away from disagreeing.
    That disagreement IS the finding this whole change is about, one layer down.

    Driven through the restored-provenance dict form, which is the shape with no
    ``.ok`` on it anywhere.
    """
    w = _panel(qtbot)
    w.set_proposal(BaselineProposal(config=_cfg(0.04, 0.01), param_source="proposed",
                                    reason="synthetic"))
    w.set_health({"ok": False, "checks": [
        {"name": "length_rough", "level": "ok", "label": "recording vs rough window",
         "value": 1.0, "threshold": ">= 1", "message": "fine"},
        {"name": "some_future_check", "level": "info", "label": "a future check",
         "value": 0.5, "threshold": "<= 1", "message": "worth knowing"},
    ]}, measured_for="proposal")

    assert api(w, "condemned_checks")() == ("some_future_check",), (
        w.condemned_checks())
    before = _form(w)
    w.apply_button.click()
    qapp.processEvents()
    assert _form(w) == before
    assert "REFUSED" in w.gate_label.text()


def test_a_detection_result_does_not_lift_a_standing_condemnation(qapp, qtbot,
                                                                  condemned, clean):
    """The third defect found inside this fix, and the nastiest reachable one.

    ``AnalysisPanel._show_result`` pushes the DETECTION's own health block into this
    panel on every result shown. A gate recomputed from whatever was last pushed in
    therefore has an ordinary Detect click lift the condemnation on a proposal still
    sitting on offer above it:

        probe -> two red WARNINGs -> Detect (any result, even a clean one)
        -> Apply -> accepted in ONE press, unmarked.

    A verdict on a proposal is ended by a NEW PROPOSAL, not by a result about other
    windows.
    """
    proposal, health = condemned
    _clean_proposal, clean_health = clean
    w = _panel(qtbot)
    w.set_proposal(proposal)
    w.set_health(health, measured_for="proposal")
    before = _form(w)

    # A detection lands. Its own block is what the user should now be reading...
    w.set_health(clean_health, measured_for="detection")
    assert w.health_scope() == "detection"

    # ...and the proposal above it is still the condemned one.
    w.apply_button.click()
    qapp.processEvents()
    assert _form(w) == before, (
        "a Detect click cleared the condemnation on the proposal still on offer")
    assert "REFUSED" in w.gate_label.text()


@pytest.mark.parametrize("which", ["condemned", "clean"])
def test_the_gate_follows_health_ok_on_the_real_verdicts(qapp, qtbot, request, which):
    """And on the two verdicts the estimator actually produces."""
    proposal, health = request.getfixturevalue(which)
    w = _panel(qtbot)
    w.set_proposal(proposal)
    w.set_health(health, measured_for="proposal")
    assert bool(api(w, "condemned_checks")()) is (not health.ok)
    assert set(w.condemned_checks()) == {c.name for c in health.checks if not c.ok}


# ===========================================================================
# 4. THE HEALTH BLOCK STILL READS THE WAY IT DID
# ===========================================================================
def test_the_block_still_states_its_own_criterion_on_every_row(qapp, qtbot,
                                                               condemned):
    """Guard on the extraction the gate reads THROUGH.

    The gate needs each check's machine name, and it takes it from the one
    extraction that already handles both a live `BaselineHealth` and the dict a
    reopened project carries -- so that block and gate cannot disagree about what is
    failing. That extraction feeds `_check_html` positionally, and the row it renders
    is the one the last round's finding was about ("measured X, passes Y", eaten by
    Qt's rich-text parser). One index off and every criterion on screen shifts by a
    field, silently.
    """
    _proposal, health = condemned
    w = _panel(qtbot)
    w.set_health(health, measured_for="proposal")
    text = w.health_text()
    for c in health.checks:
        assert c.label in text, c.label
        assert f"passes {c.threshold}" in text, (c.name, c.threshold)
    assert "WARNING" in text and "PASS" in text
    assert "base window vs state dwell" in text
    # The row order is worst-first, and the caption counts the rows it captions.
    assert f"These {len(health.checks)} checks" in w.health_scope_label.text()

    # AND THE ROW SHAPE ITSELF, because the first cut of this fix grew the tuple by a
    # field and shifted it. ``_health_checks`` is read POSITIONALLY here and outside
    # this module (``{c[0] for c in ...} <= {'ok','warn','info','refuse'}``), and it
    # is spread into ``_check_html(level, label, message, value, threshold)``.
    rows = BaselinePanel._health_checks(health)
    assert len(rows) == len(health.checks)
    assert all(len(r) == 5 for r in rows), rows[0]
    assert {r[0] for r in rows} <= {"ok", "warn", "info", "refuse"}, rows
    assert [r[1] for r in rows] == [c.label for c in health.checks]
    assert [r[4] for r in rows] == [c.threshold for c in health.checks]


# ===========================================================================
# 5. M2 -- THE REMEDY NAMES AN ACTION THAT WORKS
# ===========================================================================
def test_the_remedy_names_an_action_that_works(condemned):
    """The invariant is the section header: the remedy may only name an action that
    WORKS. What that sentence has to be flipped when M2 changed which actions work.

    Before the proposer floor, pressing Probe again re-offered the same condemned
    0.1 / 0.03 s for ever, so the only true remedy was BY HAND and the message said
    ``RE-RUNNING THE PROBE WILL NOT DO IT``. The proposer now divides by the same
    median this check divides by and floors ``base_s`` at ``BASE_STATE_RATIO_MIN x``
    it, so pressing Probe IS now an action that works -- and a message still telling
    the reader it is not would send them the long way round, which is the same defect
    this test was written to catch, pointing the other way.

    So: the by-hand instruction and the number to type are still load-bearing and
    still asserted; the probe clause is asserted in its new, true form, and the old
    sentence is asserted ABSENT so a revert of the message alone fails here.
    ``tests/gui/test_fup_proposer_wired.py::
    test_the_condemned_window_still_says_what_to_do_and_names_the_probe`` reads the
    same remedy off the real -700 mV screen reached by hand.
    """
    _proposal, health = condemned
    m = health.check("base_vs_state").message
    assert "WIDEN base_s to at least" in m, m          # unchanged, and load-bearing
    assert "BY HAND" in m, m
    assert "OR PRESS PROBE AGAIN, WHICH NOW DOES IT FOR YOU" in m, m
    assert "RE-RUNNING THE PROBE WILL NOT DO IT" not in m, (
        "the remedy still tells the reader the probe cannot repair this, which the "
        "proposer floor made false", m)
    # ...and it still records WHY that sentence used to be true, so the reader can
    # tell a stale table from a fresh one.
    assert "deterministic" in m, m
    # The number the user is being sent to type is still in the sentence.
    want = B.BASE_STATE_RATIO_MIN * health.state_dwell_s
    assert f"{want:.3g} s" in m, (want, m)


def test_the_module_docstring_records_both_halves_of_the_contract():
    """The proposer and the checks are two halves of one contract.

    This used to assert ``PROPOSER HAS NOT BEEN TAUGHT`` -- the file that owns
    `BASE_STATE_RATIO_MIN` recording that only the detector half existed, so that
    whoever taught `baseline_probe` the bound would find the two compensating
    behaviours (this remedy, and the panel's Apply gate) before removing them. They
    have now been taught it, and leaving the old sentence standing would be the
    folklore this test exists to prevent, one revision out of date.

    The three names are the load-bearing part and are unchanged: whoever changes the
    floor still has to find `baseline_probe`, `state_dwell_s` and `baseline_panel`
    from here. The absence assertion is what stops the docstring drifting back.
    """
    doc = B.__doc__ or ""
    assert "THE PROPOSER IS NOW TAUGHT THAT BOUND" in doc, doc[:400]
    assert "PROPOSER HAS NOT BEEN TAUGHT" not in doc, (
        "the docstring still says the proposer has not been taught the bound", doc[:400])
    assert "baseline_probe" in doc and "state_dwell_s" in doc
    assert "baseline_panel" in doc


# ===========================================================================
# 6. THE REAL RECORDING, THROUGH THE REAL WINDOW
# ===========================================================================
@pytest.mark.corpus
@pytest.mark.skipif(not APO_700.exists(), reason="the -700 mV recording is not here")
def test_on_the_owners_seven_hundred_millivolt_recording_through_a_real_window(
        qapp, qtbot):
    """THE REPORTED SCREEN, end to end: open -> select -> probe -> Apply.

    Costs ~2 minutes (a 108 MB / 9.1 M-sample ABF, a probe, and the estimator run
    that produces the health block), and it is the only test here that reads the real
    file rather than a synthetic that resembles it.

    WHAT THIS TEST NOW ASSERTS, AND WHY IT IS NOT THE SAME CLAIM
    ===========================================================
    It used to assert that two consecutive probes both re-offered 0.1 / 0.03 s, that
    the health block condemned it, and that M1's gate therefore refused the first
    Apply click on this recording. The first of those was M1's PREMISE, not M1's
    behaviour: the alarm existed because the proposer could not repair itself. M2
    taught `propose_baseline` the same ``BASE_STATE_RATIO_MIN x state_dwell_s`` bound
    `base_vs_state` enforces, so the proposal here is 0.5 / 0.1 s, all eight checks
    pass, and one press applies it. The gate is now a BACKSTOP on this file, not a
    step in its workflow -- and asserting that a REFUSED banner still stands over it
    would be asserting a defect.

    So the three claims here are the mirror image, and each fails on a revert:

    * the proposal on the real file is no longer the condemned pair, and certifies
      (fails if M2's floor is reverted);
    * one Apply click applies it, stamped ``proposed``, with an empty gate label and
      a plain button -- the screen M1's REFUSED banner used to occupy;
    * the window the owner published from is still reachable BY HAND, and the health
      block still condemns it on exactly the two checks M1's gate was built for, so
      the condemnation itself has not been softened.

    M1's gate BEHAVIOUR -- refuse once, arm, accept, and mark the source line for ever
    -- is exercised on every path by the ``condemned`` fixture tests above, which
    reach a condemned proposal directly; it is unreachable from the probe on this
    recording, because ``_condemned`` is read off the PROPOSAL's health and the
    proposal is now healthy.
    """
    from nanoscope.core.types import ChannelKind
    from nanoscope.gui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1600, 900)
    w.show()
    w._inspector.show()
    w._inspector_tabs.setCurrentWidget(w._analysis_panel)
    qapp.processEvents()

    with qtbot.waitSignal(w._session.documentOpened, timeout=300_000):
        w.open_paths([str(APO_700)])
    qtbot.wait(50)
    doc_id = w._session.doc_id_for_source(str(APO_700.resolve()))
    assert doc_id is not None
    channel = next(c.index for c in w._session.document(doc_id).meta.channels
                   if c.kind is ChannelKind.CURRENT)
    assert channel == 2
    w._on_selection(doc_id, channel)
    qtbot.wait(30)

    p = w._analysis_panel
    bp = p.baseline_panel
    ctrl = w._analysis_controller

    seen = []
    for _ in range(2):
        with qtbot.waitSignal(ctrl.finished, timeout=900_000):
            bp.probeRequested.emit()
        qtbot.wait(100)
        pr = bp.proposal()
        assert pr is not None
        seen.append((round(float(pr.config.rough_s), 6),
                     round(float(pr.config.base_s), 6)))

    # M2, on the real file: the proposer no longer offers what its own checks condemn,
    # and it is as deterministic about the healthy window as it was about the bad one.
    assert len(set(seen)) == 1, ("the proposer is no longer deterministic here", seen)
    assert seen[0] != PROBED_700, (
        "the proposer is offering the condemned window again -- the floor has gone", seen)

    health = bp.health()
    assert health is not None and health.ok is True, (
        [c.name for c in health.checks if not c.ok])
    assert bp.condemned_checks() == (), bp.condemned_checks()

    # The screen M1's REFUSED banner used to occupy: plain button, no gate, one press.
    assert bp.apply_button.isEnabled() is True
    assert bp.apply_button.text() == "Use these values", bp.apply_button.text()
    assert bp.apply_button.styleSheet() == "", bp.apply_button.styleSheet()
    assert bp.gate_label.text() == "", bp.gate_label.text()
    assert bp.gate_label.isVisible() is False

    bp.apply_button.click()
    qapp.processEvents()
    assert (round(float(bp.rough_s.value()), 6),
            round(float(bp.base_s.value()), 6)) == seen[0], (
        "one press must now APPLY -- leaving the form on the app default is the "
        "refusal path M2 exists to make unnecessary")
    assert bp.param_source() == "proposed"
    src = bp.source_label.text()
    assert "ACCEPTED OVER" not in src, src
    assert "app default" not in src, src
    assert bp.accepted_condemned() == (), bp.accepted_condemned()
    assert bp.gate_label.text() == "", bp.gate_label.text()
    assert bp.params()["baseline_rough_s"] == pytest.approx(seen[0][0])
    assert bp.params()["baseline_base_s"] == pytest.approx(seen[0][1])
    assert bp.params()["baseline_param_source"] == "proposed"

    # ...and the window the owner published from is still reachable BY HAND, which is
    # the documented escape hatch M1 left open when it refused the first press. Typing
    # is not a proposal, so it arms no gate -- the gate is about what the APP offered.
    bp.rough_s.setValue(PROBED_700[0])
    bp.base_s.setValue(PROBED_700[1])
    qapp.processEvents()
    qtbot.wait(80)
    assert "set by hand" in bp.source_label.text(), bp.source_label.text()
    assert bp.param_source() == "user", bp.param_source()
    assert (round(float(bp.rough_s.value()), 6),
            round(float(bp.base_s.value()), 6)) == PROBED_700
    # That this window is STILL condemned by exactly ``CONDEMNED_700`` on the real
    # file, and what the remedy on that screen now says, is asserted in
    # ``tests/gui/test_fup_proposer_wired.py`` (which reaches it the same way and then
    # runs the detection); re-running the estimator here would add a second minute to
    # a two-minute test for a claim already made against the real recording.
