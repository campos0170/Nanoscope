"""Editable Matplotlib SVG font declarations with NanoScope's fallback policy."""
from __future__ import annotations
import re
import xml.etree.ElementTree as ET
from .figure_fonts import css_font_family

ET.register_namespace("", "http://www.w3.org/2000/svg")


def editable_font_stacks(payload: bytes) -> bytes:
    root = ET.fromstring(payload)
    for node in root.iter():
        if node.tag.rsplit('}', 1)[-1] not in ('text', 'tspan'):
            continue
        style = node.get('style', '')
        family_match = re.search(r'(?:^|;)\s*font-family\s*:\s*([^;]+)', style)
        shorthand = re.search(r'(?:^|;)\s*font\s*:\s*([^;]*?)([\d.]+(?:px|pt))\s+([^;]+)', style)
        raw = node.get('font-family') or (family_match[1] if family_match else
                                          shorthand[3] if shorthand else None)
        if raw is None:
            # A tspan without its own family inherits its enclosing text declaration.
            continue
        first = re.match(r'''\s*(?:'([^']+)'|"([^"]+)"|([^,]+))''', raw)
        family = next((part for part in first.groups() if part is not None), None) if first else None
        stack = css_font_family(family)
        if family_match:
            style = style[:family_match.start()] + ';font-family: ' + stack + style[family_match.end():]
        elif shorthand:
            extra = ''
            prefix = shorthand[1].split()
            for word in prefix:
                if word in ('italic', 'oblique'):
                    extra += ';font-style: ' + word
                elif word in ('bold', 'bolder', 'lighter') or word.isdigit():
                    extra += ';font-weight: ' + word
            style = (style[:shorthand.start()] + ';font-size: ' + shorthand[2]
                     + ';font-family: ' + stack + extra + style[shorthand.end():])
        node.set('style', style.lstrip(';'))
        node.set('font-family', stack)
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)
