"""A programmatic combo setter that CANNOT fail quietly. One implementation, every window.

THE DEFECT (report Low-21)
==========================
``QComboBox.setCurrentText`` on a **non-editable** combo does nothing at all when the
text matches no item: no exception, no signal, no change. Every setter built directly
on it therefore fails silently, and the window goes on reporting the PREVIOUS
selection's numbers as if they were the requested one's.

Measured cost, twice, on this project:

* ``PlotsWindow.set_fit("gamma")`` -- one word short of ``"gamma (peaked)"`` -- produced
  a complete run of *exponential* numbers filed under gamma, and was caught only when a
  read-back check was added by hand.
* ``VDepWindow.set_metric("median depth (pA)")`` left the combo on "Capture rate (Hz)",
  and the verification pass of this very batch reported capture-rate numbers under a
  median-depth heading before noticing.

WHY THIS MODULE EXISTS RATHER THAN A HELPER IN ONE PANEL
=======================================================
The first fix put :func:`choose_combo_item` inside ``plots_panel.py`` and wired it into
**2 of that window's 7 combo setters and 0 of the sibling windows**. The second failure
above happened after that fix, in a window the fix could not reach. A rule that holds
only where it was applied is the failure mode this codebase keeps paying for, so the
rule lives in a module every panel can import and the panels' own tests assert that
none of them calls ``setCurrentText`` on a non-editable combo directly.

Imports nothing from the app but Qt: no session, no analysis, no sibling panel.
"""

from __future__ import annotations

from PySide6 import QtWidgets

__all__ = ["choose_combo_item", "set_combo_or_raise"]


def choose_combo_item(combo: QtWidgets.QComboBox, want: str, what: str) -> str:
    """Select ``want`` in ``combo``, or RAISE. Returns the label actually selected.

    THE INVARIANT: A SETTER THAT CANNOT HONOUR ITS ARGUMENT MUST NOT RETURN QUIETLY.

    Matching is exact first, then case-insensitive, then a unique case-insensitive
    prefix -- so ``"gamma"`` resolves to ``"gamma (peaked)"`` rather than raising, while
    an ambiguous or absent name raises with the valid options named. A prefix that
    matches two items is an error, not a coin toss.

    The read-back at the end defends the defence: if Qt declines the assignment for any
    reason the caller hears about it here rather than three panels later.

    Args:
        combo: the widget to select in.
        want: the label, or an unambiguous case-insensitive prefix of one.
        what: what is being set, for the message ("fit", "metric", "column"...).

    Raises:
        ValueError: ``want`` matches no item, or matches more than one by prefix.
    """
    items = [combo.itemText(i) for i in range(combo.count())]
    # Only explicitly keyed combos expose aliases. Ordinary UserRole values can be
    # arbitrary objects, and are not an invitation to select by their repr.
    key_for = getattr(combo, "selection_key", None)
    keys = [str(key_for(i)) for i in range(combo.count())] if callable(key_for) else items
    aliases = [set((label, key)) for label, key in zip(items, keys)]
    text = str(want).strip()
    low = text.lower()
    matches = [i for i, key in enumerate(keys) if key == text]
    if not matches:
        matches = [i for i, names in enumerate(aliases) if text in names]
    if not matches:
        matches = [i for i, names in enumerate(aliases) if any(name.lower() == low for name in names)]
    if not matches and low:
        matches = [i for i, names in enumerate(aliases) if any(name.lower().startswith(low) for name in names)]
    if len(matches) > 1:
        choices = [f"{items[i]} [{keys[i]}]" if keys[i] != items[i] else items[i] for i in matches]
        raise ValueError(
            f"{what} {want!r} is ambiguous: it matches {choices}. Name one exactly.")
    if not matches:
        raise ValueError(
            f"unknown {what} {want!r}; this window offers {items}. "
            f"(Setting it silently would leave the previous selection in place and the "
            f"panel would report THAT model's numbers.)")
    index = matches[0]
    combo.setCurrentIndex(index)
    if combo.currentIndex() != index or combo.currentText() != keys[index]:  # pragma: no cover
        raise RuntimeError(
            f"{what} {items[index]!r} did not take: the combo still reads "
            f"{combo.currentText()!r}")
    return items[index]



#: Alias for callers that read better as an imperative.
set_combo_or_raise = choose_combo_item
