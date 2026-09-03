"""The overlay layer, and THE ONE identity-keyed colour mapping every surface derives from.

WHY THE COLOUR MAP LIVES HERE
=============================
Before this module owned it, one recording had FOUR unrelated colours: the trace
view's layer ink (``Layer.color``, from ``LAYER_PALETTE`` by an add-order counter),
the Plots panel's source ink (``core.sources.source_colour(i)`` by DISCOVERY
POSITION), the categorical class inks (``gui.categorical``, keyed by cluster id and
so a different axis of meaning entirely), and each report window's own literals.
Two of those four move under the user's feet: a positional assignment recolours
EVERY source the moment the set or its order changes -- close one recording and
"the Zn run" is a different colour in the next picture, in a window whose whole
job is comparison.

:class:`ColourRegistry` is the fix. It is keyed by the recording's PERSISTENT key
(:func:`colour_key`), not by position, so:

* a source keeps its ink when a neighbour is closed, re-detected or reordered;
* the trace layer and the Plots panel agree on identity because layer creation and
  plot discovery ask the same registry with the same key -- ``PlotSource.key`` and
  the persistent form of ``Layer.layer_id`` are the SAME STRING
  (``"<persistent doc key>#<channel>"``), which is what makes one map across the two
  surfaces possible at all;
* a colour the user picks by hand survives a save/reopen, exactly the way
  ``Layer.color`` already did (:class:`nanoscope.core.project.ProjectState` carries
  the registry; see ``colours`` there).

IT DOES NOT BELONG IN ``StyleSettings``. That is the obvious-looking home -- the
Plots panel's style block already owns the title, the fonts and the legend
overrides -- and it is WRONG for this one. ``StyleSettings`` is per-WINDOW
presentation state: it is built fresh when the Plots window opens, thrown away
when it closes, and it is not written to the project file. A colour identity put
there would be forgotten on window close and would never reach the trace view at
all, which is half of what D1 asked for. Colour identity is per-PROJECT data about
a recording, so it lives in ``core`` beside the other per-recording state and is
persisted with it. If you are about to add a colour field to ``StyleSettings``,
this paragraph is why you should not.

WHAT THE REGISTRY STORES, AND WHAT IT DOES NOT
==============================================
Two things per key, and no ink unless the user chose one:

* a SLOT, an integer handed out in first-seen order and then never moved. The slot
  is the identity's place in a categorical cycle.
* an INK, present only when the user assigned one. A user ink wins on every
  surface -- that IS "same colour for Zn data in inspector and population plots".

The slot is rendered through a per-surface palette rather than being an ink
itself, because the two surfaces do not have the same background. A trace layer
stores the dark-pane rendering as its concrete ``Layer.color``; plot discovery
renders the registry again through its light-pane palette. The trace pane is
``#111316`` (``gui.app._apply_theme``) and Okabe-Ito's eighth member is BLACK,
which measures 1.13:1 against it -- below the 3:1 WCAG floor for a graphical object,
and darker than the pane it would be drawn on. The two
palettes are hue-aligned slot for slot (see :data:`LAYER_INKS`), so slot 3 reads as
the same colour on both surfaces without either one having to give up its own
legibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Sequence

__all__ = [
    "LAYER_INKS",
    "ViewMode",
    "Layer",
    "ColourRegistry",
    "colour_key",
]


#: *** THE TRACE VIEW'S OVERLAY INKS. Colour-vision-safe, and measured. ***
#:
#: The palette this replaced was Material Design's 300-level swatches
#: (``#4FC3F7 #FFB74D #81C784 #BA68C8 #E57373 #4DD0E1 #FFF176 #A1887F``), chosen for
#: how they look and never checked. Measured as minimum pairwise CIEDE2000 after
#: Machado-Oliveira-Fernandes (2009) severity-1.0 CVD simulation, it COLLAPSES:
#:
#:     vision          old min dE2000      new min dE2000
#:     normal              11.51               16.80
#:     deuteranopia         5.07               10.65
#:     protanopia           3.74               11.07
#:     tritanopia           3.05               10.63
#:
#: Three pairs sat under dE 10 under every one of the three deficiencies -- worst
#: was ``#4FC3F7``/``#4DD0E1`` (the sky blue and the cyan) at 3.05 under tritanopia
#: and 5.07 under deuteranopia, i.e. the first and sixth overlaid traces were
#: effectively the same colour for a deuteranope. dE 10 is the conventional "these
#: read as different colours" floor; the old palette cleared it only for normal
#: vision.
#:
#: These inks keep Okabe-Ito's HUES (within 9 degrees) and saturations (within 0.15)
#: slot for slot, so slot n here and ``OKABE_ITO[n]`` in the plot windows are the
#: same colour family -- that hue alignment is what lets ONE slot mean one colour
#: across two differently-lit surfaces. What they do NOT do is copy Okabe-Ito's
#: values: naively lifting each Okabe-Ito colour to a bright HSV value (the trick
#: ``plot.style.LEVEL_INK`` uses for a single ink) was measured at dE 4.73 under
#: deuteranopia -- WORSE than the Material palette it would have replaced, because
#: Okabe-Ito's separation is carried substantially by LIGHTNESS, and flattening
#: eight colours to one lightness for contrast against a dark pane throws that away.
#: The spread of lightness here is deliberate and is the reason the numbers hold.
#:
#: Two further constraints, both measured against the pane they are drawn on:
#:
#: * every ink clears 4.50:1 WCAG contrast on the ``#111316`` trace background
#:   (measured minimum 4.59:1; the old palette's was 5.23:1, so a little headroom was
#:   spent and none of the legibility floor was);
#: * every ink is at least dE 15 from each ink already drawn INTO that pane -- the
#:   axis/label chrome ``#b9bfc7``, the fitted-level overlay ``plot.style.LEVEL_INK``
#:   (``#24F0B8``), the event-span amber ``#FFB74D`` and the filter-preview orange
#:   ``#e69f00``. A layer the colour of the level overlay drawn ON it is a picture
#:   that lies about which line is data. That floor is enforced under NORMAL vision
#:   only, and the asymmetry is deliberate: two layer curves are told apart by colour
#:   and by NOTHING else, whereas the level overlay is a step function, the span is a
#:   filled block and the preview is announced by its own checkbox and status line --
#:   form disambiguates those where colour alone must carry the layers. Demanding the
#:   full dE-12 separation from all four reserved inks under all three deficiencies as
#:   well was tried and is INFEASIBLE inside Okabe-Ito's hues: it is eleven colours in
#:   a space that compresses to roughly two axes, and the best solve fell to dE 5.92
#:   between LAYERS -- worse than the palette being replaced.
#:
#: Slot 7 is the one that is not an Okabe-Ito hue: its Okabe-Ito member is black,
#: which this pane cannot show at all (1.13:1 -- below the 3:1 WCAG floor for a
#: graphical object, and darker than the pane), so white fills the slot instead.
LAYER_INKS: tuple[str, ...] = (
    "#23BDEB",  # blue            (Okabe-Ito #0072B2)
    "#E44802",  # vermillion      (Okabe-Ito #D55E00)
    "#10AD70",  # bluish green    (Okabe-Ito #009E73)
    "#CC98B6",  # reddish purple  (Okabe-Ito #CC79A7)
    "#A48C19",  # orange          (Okabe-Ito #E69F00)
    "#3388A1",  # sky blue        (Okabe-Ito #56B4E9)
    "#FDE26C",  # yellow          (Okabe-Ito #F0E442)
    "#FFFFFF",  # black has no dark-pane equivalent; white stands in for slot 7
)


class ViewMode(Enum):
    STACKED = "stacked"   # one pane per visible layer
    OVERLAY = "overlay"   # panes grouped by (ChannelKind, effective units) — see TraceView


@dataclass
class Layer:
    """THE UNIT OF COMPARISON — a (document, channel) pair with view state. NOT a document.

    The core workflow ("compare the -1000/-700/-600/-500 mV runs") is an operation over these,
    not over documents. Without Layer, overlay has nowhere to live but special-case branching
    inside trace_view.py, which then has to reach into every other panel. Layer makes the
    comparison the DEFAULT data path rather than a bolt-on.

    Layer membership is ALSO the authoritative include/exclude decision for a channel — which is
    why no separate `channel_enabled` map exists. ChannelInfo.is_dead only seeds the default
    layer set; the user can always add a layer for a channel flagged dead.
    """

    doc_id: str
    channel: int
    visible: bool = True
    color: str = LAYER_INKS[0]  # assigned by Session.add_layer via ColourRegistry.ink()
    time_offset_s: float = 0.0  # user alignment nudge, ADDED to meta.t0_s. Ignored when the rate
                                # is unknown (the UI disables the control).
    label: str = ""             # display label; Session fills "<file> — <channel>"
    excluded: bool = False
    # ^^ THE USER'S OWN INCLUDE/EXCLUDE DECISION — distinct from `visible`, which is merely
    # "is this the recording on screen right now". Conflating the two is what put two
    # behaviours behind one checkbox: unticking a channel had to DESTROY its layer, because
    # `Session.show_only` re-shows every layer of the recording it switches to and would
    # otherwise resurrect it the moment the user came back. Destroying it took the colour
    # and the `time_offset_s` alignment with it — persisted state, silently, without even
    # marking the project dirty (measured: dt 1.5 s and #FF00FF became 0.0 s and the next
    # palette colour on re-tick).
    #
    # With the decision recorded here instead, unticking HIDES and remembers, `show_only`
    # skips what the user excluded, and re-ticking restores exactly what was there. The two
    # fields answer two different questions and neither has to lie for the other.

    @property
    def layer_id(self) -> str:
        """Deterministic and unique: (doc_id, channel) is the key. Session forbids duplicates.

        THE SAME STRING as :attr:`nanoscope.core.sources.PlotSource.key` for the same
        recording+channel, once ``doc_id`` is the persistent key rather than the volatile
        in-memory one (which is what the project file stores, and what
        :func:`colour_key` spells). That coincidence is not a coincidence: it is what
        makes ONE colour registry serve both the trace view and the Plots panel.
        """
        return f"{self.doc_id}#{self.channel}"


# Rejected from Layer: `y_offset`. It forces a 60 MB copy per offset layer to buy a presentation
# nicety, and in the comparison workflow you *want* the baselines at their true values. Purely
# additive later.


# ================================================================================================
# Colour identity
# ================================================================================================

def colour_key(doc_key: str, channel: int | None = None) -> str:
    """THE ONE spelling of a colour identity: ``"<persistent doc key>#<channel>"``.

    ``doc_key`` must be the PERSISTENT key (a file's path, or a segment's uid — what
    :func:`nanoscope.core.segment.persistent_key` returns and what the project file
    stores), never the volatile in-memory ``doc_id``: a colour keyed by something
    that changes on reopen is a colour that does not survive reopen, which is the
    whole point of keying it at all.

    ``channel`` is ``None`` for a source that has no channel — today only the manual
    measurement table, which is one row-set for the whole session. Its key is then
    the bare ``doc_key``, which cannot collide with a channelled one because that
    form always carries a ``#``.

    Spelled here rather than at the three call sites for the reason
    :mod:`nanoscope.gui.categorical` gives about its palette: a key format written
    out twice is a second key format, and the day the two disagree the user sees two
    colours for one recording and no error anywhere.
    """
    if channel is None:
        return str(doc_key)
    return f"{doc_key}#{int(channel)}"


@dataclass
class ColourRegistry:
    """Identity → (slot, optional user ink). THE colour mapping; see the module docstring.

    Not frozen, and deliberately so: this is session-lifetime state that both the
    Session and a restore path mutate, and it is the ONE writer of a recording's
    colour. Copy it with :meth:`replace` if you need a snapshot.

    Slots are handed out in FIRST-SEEN order and then never move. That order is
    still an order — the first recording opened gets slot 0 — but it is recorded
    against the identity the first time and is stable from then on, which is
    exactly the property the positional assignment lacked: closing a neighbour, a
    re-detection, or a Files-list reorder changes nothing.

    An ink is stored ONLY when the user assigns one. The registry never
    materialises a default ink, because a default ink is a palette choice and the
    two surfaces have different palettes; see :meth:`ink`.
    """

    #: identity key → slot index. Insertion order is the assignment order.
    slots: dict[str, int] = field(default_factory=dict)
    #: identity key → the hex the USER chose. Absent means "use the slot's palette ink".
    inks: dict[str, str] = field(default_factory=dict)

    # -- reading -----------------------------------------------------------------------------
    def slot(self, key: str) -> int:
        """This identity's slot, assigning the next free one the first time it is asked.

        ASSIGNING ON READ is intentional. The alternative — an explicit ``register``
        that every surface must remember to call — is a rule that gets forgotten in
        exactly one of four callers, and the symptom is a source drawn in a colour
        nobody else knows about. Asking IS registering.

        The next slot is ``max(existing) + 1`` rather than ``len(slots)``: after a
        recording is forgotten (:meth:`forget`) reusing its number would hand a NEW
        recording the colour the user still associates with the old one.
        """
        key = str(key)
        got = self.slots.get(key)
        if got is not None:
            return got
        nxt = max(self.slots.values(), default=-1) + 1
        self.slots[key] = nxt
        return nxt

    def ink(self, key: str, palette: Sequence[str] = LAYER_INKS) -> str:
        """The colour to draw ``key`` in, on a surface whose inks are ``palette``.

        The user's assignment wins whatever the palette — that is what makes one
        recording one colour across the trace view and the Plots panel. Otherwise
        the identity's slot is rendered through this surface's palette, wrapping at
        the palette's length.

        Wrapping is a real collision past 8 identities, and it is not one a colour
        map can solve; a surface that can show that many at once should say so in
        words the way :func:`nanoscope.gui.categorical.colours_repeat` does.
        """
        key = str(key)
        chosen = self.inks.get(key)
        if chosen:
            return chosen
        if not palette:
            raise ValueError("palette must not be empty")
        return palette[self.slot(key) % len(palette)]

    def is_user_set(self, key: str) -> bool:
        """True when the colour of ``key`` is the user's own choice, not the palette's.

        The GUI needs this to offer a "reset to the palette colour" affordance that
        is honest about whether there is anything to reset.
        """
        return bool(self.inks.get(str(key)))

    def known(self, key: str) -> bool:
        """True when this identity has been seen before (READ-ONLY: does not assign)."""
        return str(key) in self.slots

    def keys(self) -> tuple[str, ...]:
        return tuple(self.slots)

    # -- writing -----------------------------------------------------------------------------
    def set_ink(self, key: str, hex_colour: str) -> None:
        """Record the user's own colour for ``key``. Also assigns its slot if new.

        No validation of ``hex_colour`` beyond non-emptiness: the value comes from a
        ``QColorDialog`` or from a project file this build wrote, and inventing a
        colour-string grammar here would be a second, weaker copy of Qt's.
        """
        key = str(key)
        hex_colour = str(hex_colour).strip()
        if not hex_colour:
            raise ValueError("a user colour must be a non-empty colour string")
        self.slot(key)
        self.inks[key] = hex_colour

    def clear_ink(self, key: str) -> None:
        """Drop the user's colour for ``key``, returning it to its slot's palette ink.

        The SLOT is kept. Forgetting it as well would hand the recording whatever
        number is free next time it is asked for, which is a different colour again
        — the reset gesture means "back to the colour this recording had", not
        "give it a new one".
        """
        self.inks.pop(str(key), None)

    def forget(self, key: str) -> None:
        """Drop everything about ``key``. For a recording removed from the project."""
        key = str(key)
        self.slots.pop(key, None)
        self.inks.pop(key, None)

    def replace(self) -> "ColourRegistry":
        """An independent copy (the dicts are not shared)."""
        return ColourRegistry(slots=dict(self.slots), inks=dict(self.inks))

    # -- persistence -------------------------------------------------------------------------
    def to_json(self) -> list[dict[str, Any]]:
        """JSON-safe rows, in slot order. The project file's ``colours`` section.

        A LIST of rows rather than a mapping because a manifest section that is a
        list is the shape every other section here already has, and because the
        order is meaningful to a human reading the file.
        """
        return [
            {"key": k, "slot": int(s), **({"ink": self.inks[k]} if k in self.inks else {})}
            for k, s in sorted(self.slots.items(), key=lambda kv: (kv[1], kv[0]))
        ]

    @classmethod
    def from_json(cls, rows: Iterable[Mapping[str, Any]] | None) -> "ColourRegistry":
        """Rebuild from :meth:`to_json`. ``None`` / missing → an empty registry.

        Tolerant on purpose, in the same spirit as
        ``plots_panel._apply_checked_keys``: a row naming a recording this session
        has not opened is KEPT, not dropped, because the recording may be reopened
        later in the same project and the user's colour for it should still be
        there. A row missing its key or carrying a non-integer slot is skipped
        rather than raising — one malformed colour must not cost the project.
        """
        reg = cls()
        for row in rows or ():
            try:
                key = str(row["key"])
                slot = int(row["slot"])
            except (KeyError, TypeError, ValueError):
                continue
            if not key:
                continue
            reg.slots[key] = slot
            ink = row.get("ink")
            if ink:
                reg.inks[key] = str(ink)
        return reg

    def __iter__(self) -> Iterator[str]:
        return iter(self.slots)

    def __len__(self) -> int:
        return len(self.slots)
