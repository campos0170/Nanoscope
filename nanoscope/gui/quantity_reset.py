"""The reset rule: when a figure's quantity changes, the words typed on it go back to automatic,
and so does the range pinned on each axis whose quantity moved.

OWNER DECISIONS 1 AND 2 (the 2 November 2026 plotting proposal, section 2 and Phase 0), and
the owner's answer on pinned ranges.
A figure's TYPED title, x label, y label and fit-label WORDING belong to the quantity it
showed when they were typed. When that quantity changes (a metric, a measurement, a plot
type, a view, an embedding), they go back to automatic. So does the PIN of an axis whose
quantity moved: the range and the major tick spacing typed in Axis properties are numbers in
that quantity's units, and over another quantity (a Y range pinned for Hz over median depths in
pA) they frame nothing. An axis whose quantity did not move keeps its pin (Voltage dependence's
x does not move with the metric), and so does another view's, unless its axis shows the same
quantity (the two pairs of Distribution summaries share their measurement). The look stays:
fonts, sizes, colours, every per-element text style, the other pins, the legend, the layout, and
the fit label's numbers, place, size and colour. Typed caption and annotation text is not touched:
the owner approved exactly "typed title, axis labels and fit-label wording" and the pins.

THE CHANGE IS ONE UNDO STEP in the window's own history
(:meth:`PlotAppearanceWidget.reset_typed_words`; Distributions' configuration history
holds it in the step of the change that caused it), words and pins together, and the window
says what went on its own status line, naming only what was cleared: "Title, Y label and Y
range reset: the plot now shows median depth." A figure with nothing typed and no pin on a moved
axis records no step and says nothing, and the sentence an earlier change left on the bar, which
may name a quantity the plot no longer shows, is taken down (:func:`announce`; a message that
is not the sentence is never touched). A label hidden with a single space is the same field and
counts as typed: it goes back to automatic with the others, so a window that draws an automatic
title or label gets it back, and one that draws no automatic title (Voltage dependence,
Subpopulations) records the step and the sentence for a hidden title with nothing new to see.
A fit label whose wording is taken back is the label nobody touched again (``fit_label``
``None`` in a ``PlotAppearance``, an empty ``style.fit_label`` in Distributions) unless its
numbers, place, size or colour were set, so Reset figure pressed next has nothing to do.

THE SENTENCE STANDS ONLY WHILE IT IS TRUE (:func:`stands`). It tells the person that what it
names is automatic, so it is taken down on the first change that brings one of those words
or ranges back -- the Undo of the reset, a title typed since, a range pinned again -- and not
on Reset figure and its Undo, nor on a change of look (nothing it names is typed then). On
the status bar the widget does it itself (``PlotAppearanceWidget`` takes down its own message
if it still shows); a window with a standing line of its own (Subpopulations, Comparison
figures) hears ``resetLapsed``; Distributions checks its configuration after every change.

The Undo and Redo tooltips of a step that did this name it (:func:`took_back`, "reset title
and Y label"), so it no longer passes for a Reset figure, which also moves the look. Distributions
has no appearance Undo: its Configuration Undo counts steps and names none.

WHAT DOES NOT RUN THE RULE. Changing the recording, or ticking another source that has the
plotted column, keeps the words (decision 2: a caption survives ticking another pore), and so
do a refresh or a re-run of the same quantity, the first draw and applying a preset (a preset
carries the look only). Distributions' column boxes offer only the columns every ticked source
has, so a source without the plotted column moves it, and the rule then runs as for a pick.
Opening a project, restoring a workspace, recalling a template and Undo and Redo move the
words themselves instead: a restore shows the words it saved, a template installs the title,
labels and wording it was saved with (none, if it was saved without), and Undo and Redo walk
the plot's own history. A project saved with a typed title over another metric opens as it was
saved; the rule acts on the NEXT change.

THE QUANTITY SELECTORS, window by window (read in the code, not assumed), and the axes each
one moves (the pin of those axes goes with the words; a view's pins are kept per view, so
leaving a view takes none off):

* Voltage dependence: the metric (the combo) moves the Y axis; the X axis is voltage whatever
  the metric. |V| moves the X axis, from the signed voltage to its magnitude and back, so it
  takes the X pin but not the words. The applied trend (Apply fit) moves no axis: a move from
  one fit to another, or to none, takes back the fit label's wording only. The first fit after
  none keeps wording typed ahead of it, as in Distributions, and so does the Apply that commits
  a recalled template's own fit while the wording is still the template's
  (``VDepWindow._fit_words_for``, :func:`promise_stands`).
* Distribution summaries: the measurement moves the value axis, which is x in ECDF and
  Survival and y in Box and Violin, so the pins of both pairs' value axes go together (one
  "value range" in the sentence); the other axis of a pair never moves. The summary type moves
  no axis: the pins are kept per pair (ECDF and Survival share theirs, as Box and Violin do),
  and a pair's pins wait for its return. Comparison figures: the figure type, X, Y, the
  measurement and the pair features move the words only; a matplotlib figure has no Axis
  properties, so no pin can be on it. Both compute only on Generate, so their controls edit a
  draft and the figure on screen keeps showing the old quantity: the words and pins go when
  ``analyze`` commits the new one, and not before.
* Event waveforms: the heatmap's X moves the map's x axis and its Y the map's y axis (the
  ensemble keeps its pins); Normalize time moves the ensemble's x axis (current stays current),
  and so does Align (start, center, extremum), which sets what t = 0 is on that axis once time is
  in seconds: measured against the alignment the drawn ensemble has, so it does nothing while
  time is normalised, when the alignment is immaterial. The view (heatmap or ensemble) moves no
  axis. Log x and Log y move the map's x and y axis too: with one ticked the map draws log10
  values on a linear axis, so they take the pin of their axis but not the words. Not when Axis
  properties changes Log scale: that dialog commits the range typed beside it, which is for the
  axis as it is about to be drawn; a range it hands back as it opened is carried across to the
  units the axis is about to have (log10 of it, or ten to the power) instead of being left in
  the old ones.
* Noise: PSD against ASD moves the PSD view's y axis (frequency stays frequency). The view
  moves no axis: going PSD, Spectrogram, PSD keeps the PSD's range.
* Subpopulations: the embedding (t-SNE or UMAP) moves both axes of the scatter, not the
  waveforms'.
* Distributions: the plot type, a column, the feature set on the PCA page, and the applied fit
  (``PlotConfig.rebased``). Its caption and the typed range of an axis that moved already went
  with the first two (the pin with the text, clause 2); the fit label's wording now goes with
  the first three, and with the applied model alone (not a first fit after none, which keeps
  wording typed ahead of it), and the sentence names the range that went. A histogram's y has no
  column of its own (a count), except as a DENSITY, which is per the unit of its column
  (``density (1/s)`` over dwell_s, plain ``density`` over frac_depth): there a new column moves
  the y axis as well. The PCA's feature set moves both axes (PC1 and PC2 are other combinations
  of other columns) and goes in the Configuration step of the tick that changed it; on any other
  plot type the same list is a clustering setting and moves nothing
  (``AnalysisIdentity.features``). The histogram's Normalize (count, density,
  probability per bin) is what its y axis counts, so a pick of it, or ``set_normalization``,
  takes the words and the y range through the same ``_rebase`` (``y_moved``); the swing to
  density that ticking a second source brings by itself (``_sync_norm_default``) runs under
  ``_updating`` and takes nothing (decision 2: the words survive a second source).
* I-V: nothing; it has one quantity (open-pore current against voltage), one fit and no
  selector. The Study figure editor: nothing; it styles a reconstructed figure, and its Panel
  box chooses which panel's words to edit, not what is plotted.

LEFT OUT ON PURPOSE. A log axis, |V|, bins, a colormap, a colour-by choice, the band, a
subset of events, Accepted only, Welch settings and a segment change how the same quantity is
drawn or which events feed it, not what is plotted: the line ``AnalysisIdentity`` already draws
between a quantity and its presentation, so they take no word back. They take no pin off either,
except where they change the numbers on an axis the pin was typed in: |V| (a magnitude for a
signed voltage) and the Event-count map's Log x and Log y (log10 values on a linear axis). A real
log axis (Voltage dependence's Log y, Noise's log-log and log frequency) keeps its range, which
is typed in the units it shows. The Distributions time unit
rescales typed limits and keeps the words, as it always has.

THE ROUTES, and which reset:

* A person's pick in the control resets, and so does the public setter that moves the same
  control (``set_metric``, ``set_x_metric``, ``set_view``, ``set_embedding``, ``set_plot_type``…):
  the control's own signal does the work, whoever moved it.
* A project restore and a template move the control with that signal blocked, or with the
  window's ``_updating`` flag up, and then draw the saved state: they never reset, and the
  words are the ones the project or the template carries. (Subpopulations' restore moves the
  embedding through the public setter, so the rule fires for a moment on a live page and its
  sentence is dropped when the saved appearance is installed.)
* A window that computes only on a button (Distribution summaries, Comparison figures) resets
  in ``analyze``, against the quantity it last committed. A template recalled there brings its
  words (and pins) with its quantity (``_words_for``: a :func:`promise`), so the Generate that
  commits it keeps them -- while they are still the template's (:func:`promise_stands`). Undo of
  the recall, or a word or pin changed after it, makes that Generate an ordinary change of
  quantity, which resets them. A project saved between the recall and the Generate does not
  carry the promise: what it holds is the project's, and the next change resets it.
* Undo and Redo only move the appearance; they never touch the quantity. They bring the words
  back over the quantity that is on screen, not the control. To return to the earlier quantity
  with its words, pick it first (nothing is typed any more, so that records no step) and Undo
  after; Undo first and the pick after is a new change, which clears the words again. In
  Distributions the Configuration Undo takes the plot type and the words back together, so
  there Undo alone is the route (picking the earlier plot type first is a step of its own).

Pure Python (``dataclasses`` only), so it can be read and tested without Qt.
"""
from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, Sequence

__all__ = ["AXIS_WORDS", "WORDS", "announce", "cleared_pins", "cleared_words", "name_list",
           "pinned_axes", "pins_of", "promise", "promise_stands", "reset_sentence", "stands",
           "took_back", "typed_words", "words_cleared", "words_of"]

#: The words a quantity change resets: (attribute, the name the status line gives it). The
#: first three are the title and the two axis labels (``None`` / ``""`` is automatic); the
#: fourth is the fit label's template (``FitLabelState.template``, ``""`` is automatic).
WORDS: tuple[tuple[str, str], ...] = (
    ("title", "Title"), ("x_label", "X label"), ("y_label", "Y label"),
    ("fit_label", "Fit label wording"))


def typed_words(source: Any, only: Sequence[str] | None = None) -> tuple[str, ...]:
    """The names of the words ``source`` holds typed text for.

    ``source`` is a :class:`~nanoscope.gui.plot_appearance.PlotAppearance` or Distributions'
    ``StyleSettings``: both carry ``title``, ``x_label``, ``y_label`` and a
    ``fit_label_state()``. A label hidden with a single space counts as typed: it is the
    same field, and Distributions has always cleared it with the others. ``only`` limits the
    answer to the named attributes (``("fit_label",)``: the wording alone); ``None`` is all.
    """
    out = []
    for field, name in WORDS:
        if only is not None and field not in only:
            continue
        text = source.fit_label_state().template if field == "fit_label" else getattr(source, field)
        if text:
            out.append(name)
    return tuple(out)


def words_of(source: Any, only: Sequence[str] | None = None) -> tuple[str, ...]:
    """The words ``source`` holds, as text, in :data:`WORDS` order (``""`` is automatic).

    Same ``source`` and ``only`` as :func:`typed_words`, but the values rather than the names.
    """
    out = []
    for field, _name in WORDS:
        if only is not None and field not in only:
            continue
        text = source.fit_label_state().template if field == "fit_label" else getattr(source, field)
        out.append(text or "")
    return tuple(out)


def cleared_words(appearance: Any,
                  only: Sequence[str] | None = None) -> tuple[Any, tuple[str, ...]]:
    """``(appearance with its typed words back to automatic, the names that were cleared)``.

    Only the words change: every other field, including the fit label's numbers, place,
    size and colour and every text style, is carried over untouched. Nothing typed means
    the same appearance back and no names. ``only`` limits the words as in :func:`typed_words`.
    """
    names = typed_words(appearance, only)
    if not names:
        return appearance, ()
    changes: dict[str, Any] = {}
    for field in ("title", "x_label", "y_label"):
        if getattr(appearance, field) and (only is None or field in only):
            changes[field] = None
    label = appearance.fit_label
    if label is not None and label.template and (only is None or "fit_label" in only):
        # A label with nothing left to say is the label nobody touched (None), not one holding an
        # empty template: they draw the same, but only None leaves the appearance equal to the
        # default one, so a Reset figure pressed next is a no-op, not a step that changes nothing.
        cleared = replace(label, template="")
        changes["fit_label"] = None if cleared == type(label)() else cleared
    return replace(appearance, **changes), names


#: What the status line calls an axis, by side: the two a window draws a range on.
AXIS_WORDS: dict[str, str] = {"bottom": "X", "left": "Y"}


def _axis(entry: Sequence[str]) -> tuple[str, str, str]:
    """``(view, side, name)`` of one axis a window names. ``view`` is the one whose pin it
    is (``""`` in a window with one view, as ``PlotAppearance.for_view``); the name is the
    status line's word for the axis and defaults to its side's ("X", "Y")."""
    view, side, *name = entry
    return view, side, name[0] if name else AXIS_WORDS[side]


def pinned_axes(appearance: Any, axes: Sequence[Sequence[str]],
                primary: str = "") -> tuple[str, ...]:
    """The names of the pins ``appearance`` holds on ``axes``, as the status line says them:
    "Y range" for a range, "Y major tick" for a tick spacing, each once, in the order given.

    ``axes`` are the axes whose quantity moved, ``(view, side)`` or ``(view, side, name)``;
    ``primary`` is the window's primary view (``PlotAppearance.for_view``), whose pin a bare
    side key may be. Two axes given the same name (Distribution summaries' value axis, x in
    one pair and y in the other) are said once.
    """
    out: list[str] = []
    for entry in axes:
        view, side, name = _axis(entry)
        pin = appearance.for_view(view, primary=primary).axis_pin(side)
        if pin.range() is not None:
            out.append(f"{name} range")
        if pin.tick_major is not None:
            out.append(f"{name} major tick")
    return tuple(dict.fromkeys(out))


def cleared_pins(appearance: Any, axes: Sequence[Sequence[str]],
                 primary: str = "") -> tuple[Any, tuple[str, ...]]:
    """``(the appearance with the pins of ``axes`` taken off, the names of what was taken)``.

    The range and major tick of each axis go (``PlotAppearance.without_axis_pin``) and
    nothing else does. No pin on those axes means the same appearance back and no names.
    """
    names = pinned_axes(appearance, axes, primary)
    for entry in axes:
        view, side, _name = _axis(entry)
        appearance = appearance.without_axis_pin(view, side, primary=primary)
    return appearance, names


def pins_of(appearance: Any, axes: Sequence[Sequence[str]],
            primary: str = "") -> tuple[tuple[Any, Any], ...]:
    """The range and major tick ``appearance`` pins on each of ``axes``, as values (same
    ``axes`` and ``primary`` as :func:`pinned_axes`, which gives their names)."""
    out = []
    for entry in axes:
        view, side, _name = _axis(entry)
        pin = appearance.for_view(view, primary=primary).axis_pin(side)
        out.append((pin.range(), pin.tick_major))
    return tuple(out)


def promise(quantity: Any, appearance: Any, only: Sequence[str] | None = None,
            axes: Sequence[Sequence[str]] = (), primary: str = "") -> tuple[Any, ...]:
    """What a recalled template promises the commit that draws its ``quantity``: the words
    (and the pins of ``axes``) it has just installed in ``appearance`` belong to that quantity.

    A window that computes only on a button (Distribution summaries, Comparison figures) and
    Voltage dependence's Apply fit keep this beside the draft; ``only``, ``axes`` and ``primary``
    are what that window's rule resets (:func:`typed_words`, :func:`pinned_axes`).
    """
    return quantity, words_of(appearance, only), pins_of(appearance, axes, primary)


def promise_stands(promised: Any, quantity: Any, appearance: Any,
                   only: Sequence[str] | None = None, axes: Sequence[Sequence[str]] = (),
                   primary: str = "") -> bool:
    """Whether the commit of ``quantity`` is the one a recalled template promised for, and the
    words and pins on the figure are still exactly the ones it installed.

    ``promised`` is a :func:`promise`, or ``None`` (nothing recalled, or it was spent). Undo
    of the recall, or a word or pin changed after it, leaves on the figure what was typed
    or pinned for the quantity it showed until now: the commit is then an ordinary change of
    quantity and the rule runs.
    """
    return promised is not None and promised == promise(quantity, appearance, only, axes, primary)


def words_cleared(before: Any, after: Any) -> tuple[str, ...]:
    """The names typed in ``before`` and no longer typed in ``after``: what a rule that
    rewrites a whole configuration (Distributions') actually cleared."""
    gone = set(typed_words(before)) - set(typed_words(after))
    return tuple(name for _field, name in WORDS if name in gone)


def stands(took: Sequence[str], now: Sequence[str]) -> bool:
    """Whether the sentence that said ``took`` was reset still describes the figure.

    The sentence tells the person those words and ranges are automatic, so it is true only
    while none of them is typed or pinned again. ``now`` is what the figure holds now:
    :func:`typed_words` and :func:`pinned_axes` of its appearance (Distributions: of its style,
    and of ``AxesConfig.ranges``). It stops standing on the first change that brings one back --
    the Undo of the reset, a title typed since, a range pinned again -- and not on a change of
    look, nor on Reset figure or its Undo, after which nothing is typed or pinned either.
    """
    return not set(took) & set(now)


def reset_sentence(names: Sequence[str], shows: str, *, lead: str = "the plot now shows") -> str:
    """"Title and Y label reset: the plot now shows median depth." -- names only what was cleared.

    ``names`` come from :func:`cleared_words` / :func:`words_cleared`; ``shows`` is the
    window's phrase for the new quantity ("median depth", "the spectrogram"). A window whose
    picture follows the control only when its button is pressed (Distributions) passes a
    ``lead`` that is true meanwhile. Empty ``names`` give an empty string: nothing was
    cleared, so there is nothing to say.
    """
    names = tuple(names)
    if not names:
        return ""
    # Later names read in running text ("fit label wording"), but "X label" keeps its capital.
    spoken = [names[0]] + [n if n[1:2] == " " else n[0].lower() + n[1:] for n in names[1:]]
    listed = spoken[0] if len(spoken) == 1 else ", ".join(spoken[:-1]) + " and " + spoken[-1]
    return f"{listed} reset: {lead} {shows}."


def name_list(names: Sequence[str]) -> str:
    """"title, Y label and Y range": ``names`` in running text, none of them first in a sentence
    (the Undo tooltip's "reset title, Y label and Y range"). A name keeps its capital only where
    it belongs, as in :func:`reset_sentence`."""
    spoken = [n if n[1:2] == " " else n[0].lower() + n[1:] for n in names]
    return spoken[0] if len(spoken) == 1 else ", ".join(spoken[:-1]) + " and " + spoken[-1]


def took_back(before: Any, after: Any) -> tuple[str, ...]:
    """The names a step from ``before`` to ``after`` took back to automatic -- ``("Title",
    "Y label", "Y range")`` -- when it did that and nothing else; ``()`` for any other step.

    That is what a change of quantity records (``PlotAppearanceWidget.reset_typed_words``):
    typed words back to automatic and the range and major tick of some pinned axes taken off,
    the look untouched. Read off the two values (the history holds nothing else), it lets the
    Undo and Redo tooltips say what the step does instead of passing for a Reset figure, which
    also moves the look. Two steps with the same pair of values cannot be told apart: a Reset
    figure pressed on a figure whose only customisation is typed words and pins reads the same.
    """
    moved = {f.name for f in fields(before) if getattr(before, f.name) != getattr(after, f.name)}
    if not moved or not moved <= {field for field, _name in WORDS} | {"axes"}:
        return ()
    expected, names = cleared_words(before, tuple(f for f in moved if f != "axes"))
    pins: list[str] = []
    if "axes" in moved:
        keys = {key for key, _ in (before.axes or ())} | {key for key, _ in (after.axes or ())}
        for key in sorted(key for key in keys if before.axis_pin(key) != after.axis_pin(key)):
            view, _dot, side = key.rpartition(".")
            if side not in AXIS_WORDS:
                return ()
            held = before.axis_pin(key)
            expected = expected.without_axis_pin(view, side)
            pins += ([f"{AXIS_WORDS[side]} range"] if held.range() is not None else [])
            pins += ([f"{AXIS_WORDS[side]} major tick"] if held.tick_major is not None else [])
    return tuple(dict.fromkeys([*names, *pins])) if expected == after else ()


def announce(bar: Any, said: str, earlier: str, *, msecs: int = 10000) -> str:
    """Show ``said`` on the status bar ``bar``; with nothing to say, take ``earlier`` off it.

    A quantity change reached the rule, so the sentence an earlier change left on the bar
    ("the plot now shows median depth" over an event-count plot) no longer describes the plot:
    it goes, but only if the bar still shows it, so a message that is not the sentence (or one
    that has timed out) is left alone. Returns the sentence now standing, ``""`` for none.
    ``bar`` is a ``QStatusBar``, used by method name only, so this module stays free of Qt.
    """
    if said:
        bar.showMessage(said, msecs)
    elif earlier and bar.currentMessage() == earlier:
        bar.clearMessage()
    return said
