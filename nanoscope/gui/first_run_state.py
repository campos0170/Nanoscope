"""Where the first-run claim sits in the settings store, and how to read it.

``gui.app.main`` makes the claim on a brand-new store (``first_analysis_guide.claim_first_run``)
and the Detection panel reads it to start in Essential. The panel and the guide window are
modules that may not import each other, so what both must agree on lives here: the key, the two
values it takes, and how to read them (the control level the claim stands for, and whether the
guide is still due).

Plain Python over a ``QSettings`` that is handed in: no widget, and nothing imported from ``gui``,
which is what lets a panel import it.
"""

from __future__ import annotations

__all__ = ["FIRST_RUN_DONE", "FIRST_RUN_KEY", "FIRST_RUN_PENDING",
           "first_run_control_level", "first_run_guide_due"]

#: The settings key of the claim. Absent means no claim was made. ``claim_first_run`` writes it
#: (and takes it back if the store cannot keep it) and ``finish_first_run`` rewrites it; nothing
#: else does.
FIRST_RUN_KEY = "onboarding/first_run_v1"
#: Claimed on a brand-new store, and the Guided first analysis has not opened yet. Written before
#: the window is built, so a launch that dies before the guide opens leaves the next launch to
#: open it.
FIRST_RUN_PENDING = "pending"
#: The guide has opened: the first run is over. The value is also the control level the claim
#: stands for, as it was when the claim was a single step.
FIRST_RUN_DONE = "essential"


def first_run_guide_due(settings) -> bool:
    """True while the claim is pending: the first run was claimed and the guide has not opened."""
    return str(settings.value(FIRST_RUN_KEY, "")) == FIRST_RUN_PENDING


def first_run_control_level(settings, default: str) -> str:
    """The control level the claim stands for, or ``default`` when no claim is held.

    "essential" from the moment of the claim (pending) and after the guide has opened (done): the
    Detection panel asks this last, after the user's own saved choices.
    """
    claimed = str(settings.value(FIRST_RUN_KEY, "")) in (FIRST_RUN_PENDING, FIRST_RUN_DONE)
    return "essential" if claimed else default
