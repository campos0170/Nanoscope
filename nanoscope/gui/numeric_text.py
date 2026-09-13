"""Strict ASCII-only numeric text parsing, shared by every free-text numeric field.

``float()`` is more permissive than the ASCII literal a user actually typed: it accepts
any Unicode decimal-digit script via CPython's digit normalisation (``float("١٢٠")`` is
``120.0``), and PEP 515 underscore grouping (``float("1_20")`` is also ``120.0``). Both
let a value nobody meant to type commit silently instead of being refused like ``"abc"``
is. Beta-sweep B27 (2026-09-12) found this independently reproduced in three boxes —
the open-pore reference's current field, the notch-line list, and the acquisition
bandwidth dialog's cutoff — each with its own ad hoc ``"_" in text`` guard that caught
the underscore but not the digit script, because bare ``float()`` was still the parser
underneath. One strict parser, used everywhere a user types a bare number, closes the
whole class rather than the one token each guard happened to test for.
"""
from __future__ import annotations

import math

#: The single character this module allows OUTSIDE plain ASCII: the Unicode MINUS SIGN
#: (U+2212), which some boxes accept as a documented convenience for a pasted signed
#: value (see ``allow_unicode_minus``). Nothing else non-ASCII is ever accepted.
_UNICODE_MINUS = "−"


def parse_strict_number(text: object, *, allow_unicode_minus: bool = False) -> float | None:
    """Parse ``text`` as a finite float using ONLY the ASCII numeral grammar a user
    actually typed. Returns ``None`` for anything that is not a finite number typed
    that way; never raises.

    Deliberately NOT ``float(text)`` alone: that accepts PEP 515 underscore grouping
    and any Unicode decimal-digit script, silently turning ``"1_20"`` or ``"١٢٠"`` into
    ``120.0``. This parser requires the text to be pure ASCII (after the one documented
    minus-sign normalisation below) before ever calling ``float()``, so no digit script
    other than plain ASCII 0-9 can reach it.

    ``allow_unicode_minus``: normalise a single LEADING U+2212 MINUS SIGN to ASCII
    ``-`` first. Some boxes offer this as a convenience for a pasted signed value
    (e.g. copied from a plot's axis or another application) and document it; it
    relaxes nothing else, including a minus sign anywhere but the front.
    """
    s = str(text).strip()
    if allow_unicode_minus and s.startswith(_UNICODE_MINUS):
        s = "-" + s[len(_UNICODE_MINUS):]
    if not s or not s.isascii() or "_" in s:
        return None
    try:
        value = float(s)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None
