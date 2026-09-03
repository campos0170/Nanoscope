"""The ONE identity-keyed colour map, and the measurement that says the trace palette is safe.

Two questions, both of which the app previously answered by accident:

**D1 — is one recording one colour?** It was not. Four independent assignments drew
the same recording (the trace view's ``Layer.color``, the Plots panel's positional
``source_colour(i)``, the categorical class map, and each report window's literals),
and two of them moved whenever the set of open recordings changed.
:class:`~nanoscope.core.layer.ColourRegistry` is the single map; these tests pin the
properties that make it worth having — assignment by identity, stability under every
change that used to repaint, the user's ink winning on every surface, and a
round-trip through the project file.

**D2 — can a colour-blind reader tell two overlaid traces apart?** The old
``LAYER_PALETTE`` (Material 300s) could not: measured below, three pairs collapsed
under every one of the three common deficiencies. The CVD measurement is IN THIS FILE
rather than in a comment because a palette is edited by eye, and by eye is exactly how
it got there.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from nanoscope.core.layer import LAYER_INKS, ColourRegistry, Layer, colour_key


# ================================================================================================
# colour_key — one spelling of an identity
# ================================================================================================

def test_the_colour_key_is_the_layer_id_spelled_once():
    """``PlotSource.key`` and ``Layer.layer_id`` are the SAME STRING for one recording.

    That coincidence is the load-bearing one: it is what lets a single registry serve
    the trace view and the Plots panel. If this ever fails, the two surfaces are asking
    the map different questions and will answer the user with two colours.
    """
    assert colour_key("/data/zn_30h.abf", 2) == Layer("/data/zn_30h.abf", 2).layer_id


def test_a_channelless_source_keys_on_the_bare_document_key():
    """The manual table has no channel. Its key cannot collide with a channelled one,
    because that form always carries a '#'."""
    assert colour_key("manual-measurements") == "manual-measurements"
    assert "#" not in colour_key("manual-measurements")


# ================================================================================================
# ColourRegistry
# ================================================================================================

def test_slots_are_assigned_on_first_ask_and_never_move():
    reg = ColourRegistry()
    assert reg.slot("a#0") == 0
    assert reg.slot("b#0") == 1
    assert reg.slot("a#0") == 0          # asking again does not re-assign
    assert reg.slot("c#0") == 2


def test_a_forgotten_slot_is_not_handed_to_the_next_recording():
    """Reusing the number would give a NEW recording the colour the user still
    associates with the one that left — which is the positional failure again, wearing
    a different hat."""
    reg = ColourRegistry()
    reg.slot("a#0"), reg.slot("b#0")
    reg.forget("a#0")
    assert reg.slot("c#0") == 2, "slot 0 was recycled onto a different recording"


def test_the_same_slot_renders_through_each_surfaces_own_palette():
    """The registry stores a SLOT, not an ink, because the two surfaces are not lit the
    same. Slot 1 is slot 1 in both; the ink each surface draws is its own."""
    from nanoscope.core.sources import SOURCE_COLOURS

    reg = ColourRegistry()
    reg.slot("a#0")
    assert reg.ink("a#0", LAYER_INKS) == LAYER_INKS[0]
    assert reg.ink("a#0", SOURCE_COLOURS) == SOURCE_COLOURS[0]


def test_a_user_ink_overrides_every_palette():
    reg = ColourRegistry()
    reg.set_ink("a#0", "#FF00FF")
    assert reg.ink("a#0", LAYER_INKS) == "#FF00FF"
    assert reg.ink("a#0", ("#111111", "#222222")) == "#FF00FF"
    assert reg.is_user_set("a#0")


def test_clearing_a_user_ink_returns_the_recording_to_ITS_slot():
    """Not to a new one. "Reset" means "the colour this recording had", and dropping the
    slot as well would hand it whatever number is free next."""
    reg = ColourRegistry()
    reg.slot("a#0"), reg.slot("b#0")
    reg.set_ink("b#0", "#FF00FF")
    reg.clear_ink("b#0")
    assert not reg.is_user_set("b#0")
    assert reg.ink("b#0", LAYER_INKS) == LAYER_INKS[1]


def test_inks_wrap_at_the_palette_length():
    reg = ColourRegistry()
    for i in range(len(LAYER_INKS) + 1):
        reg.slot(f"r{i}#0")
    assert reg.ink(f"r{len(LAYER_INKS)}#0", LAYER_INKS) == LAYER_INKS[0]


def test_an_empty_palette_is_refused_rather_than_indexed():
    with pytest.raises(ValueError):
        ColourRegistry().ink("a#0", ())


def test_known_does_not_assign():
    """A read that silently registers would make "has the user seen this?" unanswerable."""
    reg = ColourRegistry()
    assert not reg.known("a#0")
    assert len(reg) == 0


def test_a_blank_user_colour_is_refused():
    with pytest.raises(ValueError):
        ColourRegistry().set_ink("a#0", "   ")


def test_the_registry_round_trips_through_json():
    reg = ColourRegistry()
    reg.slot("a#0"), reg.slot("b#1")
    reg.set_ink("b#1", "#FF00FF")
    back = ColourRegistry.from_json(reg.to_json())
    assert back.slots == reg.slots and back.inks == reg.inks


def test_a_row_naming_an_unopened_recording_is_kept_not_dropped():
    """The recording may be reopened later in the same project, and the user's colour
    for it must still be there when it is. Same policy as ``_apply_checked_keys``."""
    reg = ColourRegistry.from_json([{"key": "/gone/pore.abf#0", "slot": 3, "ink": "#FF00FF"}])
    assert reg.ink("/gone/pore.abf#0", LAYER_INKS) == "#FF00FF"


def test_a_malformed_row_is_skipped_rather_than_costing_the_project():
    reg = ColourRegistry.from_json(
        [{"slot": 0}, {"key": "", "slot": 1}, {"key": "a#0", "slot": "two"},
         {"key": "b#0", "slot": 5}])
    assert reg.keys() == ("b#0",)


def test_from_json_of_nothing_is_an_empty_registry():
    assert len(ColourRegistry.from_json(None)) == 0
    assert len(ColourRegistry.from_json([])) == 0


def test_replace_does_not_share_its_dicts():
    reg = ColourRegistry()
    reg.set_ink("a#0", "#FF00FF")
    copy = reg.replace()
    copy.set_ink("a#0", "#00FF00")
    assert reg.ink("a#0") == "#FF00FF"


# ================================================================================================
# D2 — the trace palette, measured
# ================================================================================================
#
# Machado, Oliveira & Fernandes (2009) severity-1.0 CVD matrices — the same ones the
# Chrome and Firefox devtools vision simulators use — followed by CIEDE2000 in CIE Lab.
# ~50 lines of arithmetic in a test file is a lot, and it is the point: a palette is
# edited BY EYE, and by eye is how the Material swatches this replaced got in. The
# numbers below are what "colourblind-safe" means here, and they fail if someone
# retouches an ink.

_CVD = {
    "normal": np.eye(3),
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                              [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                            [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779],
                            [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}
_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                 [0.2126729, 0.7151522, 0.0721750],
                 [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])

#: The trace pane's background (``gui.app._apply_theme`` sets pyqtgraph's global to it).
PANE_BG = "#111316"
#: Everything else already drawn INTO that pane. A layer the colour of one of these is a
#: picture that lies about which line is the data.
RESERVED = {
    "axis/label chrome": "#b9bfc7",       # gui.app pg foreground
    "fitted level overlay": "#24F0B8",    # plot.style.LEVEL_INK
    "event span": "#FFB74D",              # gui.overlays default span colour
    "filter preview": "#e69f00",          # gui.trace_view.FILTER_PREVIEW_HEX
}
#: The palette this replaced: Material Design 300s, picked by eye and never measured.
_OLD_LAYER_PALETTE = ("#4FC3F7", "#FFB74D", "#81C784", "#BA68C8",
                      "#E57373", "#4DD0E1", "#FFF176", "#A1887F")
#: The conventional "these read as two colours" floor for CIEDE2000.
_DISTINCT = 10.0


def _rgb(hex_colour: str) -> np.ndarray:
    h = hex_colour.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(rgb: np.ndarray) -> np.ndarray:
    xyz = _XYZ @ _linear(rgb) / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def _lab_as_seen(hex_colour: str, vision: str) -> np.ndarray:
    return _lab(np.clip(_CVD[vision] @ _rgb(hex_colour), 0.0, 1.0))


def _relative_luminance(hex_colour: str) -> float:
    c = _linear(_rgb(hex_colour))
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def _contrast(hex_colour: str, background: str = PANE_BG) -> float:
    a, b = _relative_luminance(hex_colour), _relative_luminance(background)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _de2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """CIEDE2000. Written out because the alternative is a new third-party dependency
    for one number in one test."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1, C2 = float(np.hypot(a1, b1)), float(np.hypot(a2, b2))
    Cbar = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7))) if Cbar > 0 else 0.5
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = float(np.hypot(a1p, b1)), float(np.hypot(a2p, b2))
    h1p = float(np.degrees(np.arctan2(b1, a1p)) % 360)
    h2p = float(np.degrees(np.arctan2(b2, a2p)) % 360)
    dLp, dCp = L2 - L1, C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    else:
        dhp = h2p - h1p - 360 if h2p - h1p > 180 else h2p - h1p + 360
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    else:
        hbp = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = (-np.sin(np.radians(2 * 30 * np.exp(-(((hbp - 275) / 25) ** 2))))
          * 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7)))
    return float(np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                         + Rt * (dCp / Sc) * (dHp / Sh)))


def _worst_pair(palette, vision: str) -> tuple[float, str, str]:
    labs = [_lab_as_seen(h, vision) for h in palette]
    return min(((_de2000(labs[i], labs[j]), palette[i], palette[j])
                for i, j in itertools.combinations(range(len(palette)), 2)),
               key=lambda t: t[0])


@pytest.mark.parametrize("vision", ["normal", "deuteranopia", "protanopia", "tritanopia"])
def test_every_pair_of_layer_inks_is_distinguishable_under_every_vision(vision):
    """THE D2 GUARANTEE, and the numbers that back it.

    Measured minimum pairwise CIEDE2000, old palette -> new:

        normal          11.51 -> 16.80
        deuteranopia     5.07 -> 10.65
        protanopia       3.74 -> 11.07
        tritanopia       3.05 -> 10.63

    The old palette's worst pair was ``#4FC3F7``/``#4DD0E1`` — the FIRST and SIXTH
    overlaid traces — at dE 3.05 under tritanopia and 5.07 under deuteranopia. Three
    pairs sat under the dE-10 floor under each of the three deficiencies, so the
    overlay comparison this app is built around was unreadable for roughly 1 in 12 men.
    """
    worst, one, two = _worst_pair(LAYER_INKS, vision)
    assert worst >= _DISTINCT, (
        f"{one} and {two} are dE2000 {worst:.2f} apart under {vision}; "
        f"{_DISTINCT} is the floor at which two colours read as two colours")


def test_the_palette_that_was_replaced_really_did_fail_this(  ):
    """The control. If the old swatches passed, the change was cosmetic and this whole
    file is theatre — so the test that says they failed is worth as much as the one
    that says the new ones pass."""
    assert _worst_pair(_OLD_LAYER_PALETTE, "normal")[0] >= _DISTINCT
    for vision in ("deuteranopia", "protanopia", "tritanopia"):
        assert _worst_pair(_OLD_LAYER_PALETTE, vision)[0] < _DISTINCT


def test_every_layer_ink_is_legible_on_the_dark_trace_pane():
    """A colour-safe palette nobody can see is not an improvement. 4.5:1 is WCAG AA for
    normal text; a 1 px curve is thinner than text, so this is a floor and not a target.

    It is also the constraint that rules out using ``OKABE_ITO`` unchanged here:
    its eighth member is BLACK, which measures 1.13:1 against this pane. 3:1 is the
    WCAG floor for a graphical object; 1.13 is "the same colour as the background",
    and it is darker than it, so the eighth overlaid recording would simply be absent.
    """
    from nanoscope.plot.style import OKABE_ITO

    for ink in LAYER_INKS:
        assert _contrast(ink) >= 4.5, f"{ink} is {_contrast(ink):.2f}:1 on {PANE_BG}"
    assert _contrast(OKABE_ITO[7]) < 1.5, "the reason slot 7 is not Okabe-Ito's black"


def test_no_layer_ink_collides_with_something_already_drawn_in_that_pane():
    """The trace pane is not empty. A layer the colour of the fitted-level overlay, of
    the event-span shading or of the filter-preview ghost drawn ON it makes the picture
    ambiguous in the one way that matters — which line is the recording?

    dE 15 rather than the dE-10 "distinguishable" floor because these are drawn
    OVERLAPPING, not side by side.

    NORMAL VISION ONLY, and the asymmetry is the point rather than a gap. Two layer
    curves are told apart by colour and by nothing else, which is why the test above
    holds them to dE 10 under every deficiency. These four are told apart by FORM as
    well -- the level overlay is a step function, the span is a filled block, the
    preview is announced by its own checkbox and status line -- so colour carries less
    of the load. Requiring the full separation from all four under all three
    deficiencies too was solved for and is infeasible inside Okabe-Ito's hues: it is
    eleven colours in a space that compresses to about two axes, and the best solution
    put two LAYERS dE 5.92 apart, i.e. worse than the palette being replaced.
    """
    for name, reserved in RESERVED.items():
        for ink in LAYER_INKS:
            d = _de2000(_lab(_rgb(ink)), _lab(_rgb(reserved)))
            assert d >= 15.0, f"{ink} is dE {d:.1f} from the {name} ({reserved})"


def test_the_layer_inks_stay_hue_aligned_with_the_plot_palette():
    """WHY THE TWO SURFACES CAN SHARE ONE SLOT. Slot n here and ``OKABE_ITO[n]`` in the
    plot windows are the same colour FAMILY, so "the Zn run is the green one" survives
    the walk from the trace view to the population plot even when neither surface can
    use the other's exact ink.

    Slot 7 is exempt and says so: Okabe-Ito's eighth is black, which this pane cannot
    show at all (see the contrast test), so white stands in for it.

    9 degrees, measured worst 8.2 (slot 4, the orange). The bound is what the palette
    was SOLVED under, not a value fitted to the answer afterwards: widening it buys
    separability at the cost of the cross-surface recognition this whole design is for.
    """
    import colorsys

    from nanoscope.plot.style import OKABE_ITO

    for slot, (mine, theirs) in enumerate(zip(LAYER_INKS[:7], OKABE_ITO[:7])):
        h_mine = colorsys.rgb_to_hsv(*_rgb(mine))[0] * 360
        h_theirs = colorsys.rgb_to_hsv(*_rgb(theirs))[0] * 360
        drift = min(abs(h_mine - h_theirs), 360 - abs(h_mine - h_theirs))
        assert drift <= 9.0, (
            f"slot {slot}: {mine} is {drift:.1f} deg from {theirs}; the two surfaces stop "
            f"reading as the same colour")


def test_a_naive_value_lift_of_okabe_ito_would_have_been_worse():
    """The obvious cheap fix, measured and rejected — recorded so nobody re-proposes it.

    ``plot.style.LEVEL_INK`` lifts ONE Okabe-Ito colour to full HSV value to make it
    read on a dark pane, and doing the same to all eight is the natural next thought.
    It measures dE 4.73 under deuteranopia — WORSE than the Material palette it would
    have replaced (5.07) — because Okabe-Ito's separation is carried substantially by
    LIGHTNESS, and flattening eight colours to one lightness throws that away.
    """
    import colorsys

    from nanoscope.plot.style import OKABE_ITO

    lifted = []
    for i, hex_colour in enumerate(OKABE_ITO):
        if i == 7:                      # black has no lift; the same neutral stand-in
            lifted.append("#C8C8C8")
            continue
        h, s, _v = colorsys.rgb_to_hsv(*_rgb(hex_colour))
        r, g, b = colorsys.hsv_to_rgb(h, s, 0.88)
        lifted.append("#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255)))

    naive = _worst_pair(lifted, "deuteranopia")[0]
    material = _worst_pair(_OLD_LAYER_PALETTE, "deuteranopia")[0]
    assert naive < material, "the lift stopped being worse than what it replaced; re-measure"
    assert _worst_pair(LAYER_INKS, "deuteranopia")[0] > 2 * naive
