"""Offline Markdown fallback using the configured bundled Node/Marked runtime.

Python-Markdown remains the preferred backend in gen_page.py. This adapter keeps
the manual's existing ASCII heading anchors when that optional module is absent.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

BACKEND = "bundled Marked with compatible heading anchors"


def _runtime():
    dependencies = Path(sys.executable).parent.parent.parent
    node = Path(os.environ.get("NANOSCOPE_MANUAL_NODE", str(dependencies / "node/bin/node")))
    module = Path(os.environ.get("NANOSCOPE_MANUAL_MARKED", str(dependencies / "node/node_modules/marked/lib/marked.esm.js")))
    if not node.is_file() or not module.is_file():
        raise RuntimeError("Install Python-Markdown, or configure NANOSCOPE_MANUAL_NODE and NANOSCOPE_MANUAL_MARKED for the offline bundled backend")
    return node, module


def _render(text, *, headings):
    node, module = _runtime()
    bridge = Path(__file__).with_name("render_markdown.mjs")
    result = subprocess.run([str(node), str(bridge), str(module)],
                            input=json.dumps({"source": str(text)}), text=True,
                            capture_output=True, check=True, timeout=30)
    body = json.loads(result.stdout)["html"]
    if not headings:
        return body
    ids = set(re.findall(r'\bid="([^"]+)"', body))

    def heading(match):
        level, content = match.groups()
        name = html.unescape(re.sub(r"<[^>]+>", "", content))
        name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        name = re.sub(r"[^\w\s-]", "", name).strip().lower()
        anchor = re.sub(r"[-\s]+", "-", name)
        while anchor in ids or not anchor:
            numbered = re.match(r"^(.*)_([0-9]+)$", anchor)
            anchor = (f"{numbered[1]}_{int(numbered[2]) + 1}" if numbered else anchor + "_1")
        ids.add(anchor)
        return f'<h{level} id="{anchor}">{content}</h{level}>'

    return re.sub(r"<h([1-6])>(.*?)</h\1>", heading, body, flags=re.S)


def markdown(text):
    return _render(text, headings=False)


class Markdown:
    def __init__(self, extensions=(), extension_configs=None):
        supported = {"tables", "fenced_code", "sane_lists", "toc"}
        if set(extensions) - supported:
            raise ValueError("The offline manual backend received an unsupported extension")
        self.headings = "toc" in extensions

    def convert(self, text):
        return _render(text, headings=self.headings)
