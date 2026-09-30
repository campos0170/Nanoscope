"""Build the control-reference section of the NanoScope manual from `inventory.json`.

This is half of a two-step build; `gen_page.py` is the other half and calls into this
module. Nothing here reads or writes anything outside `docs/`:

    docs/inventory.json          in   — the maintained control inventory (counted at build time)
    docs/_build/reference.html   out  — the rendered section, for inspection only

The page build does not need the file on disk: `gen_page.py` imports
`build_reference()` directly. Run this script on its own only when you want to look at
the reference section by itself, or to check that a re-taken inventory still parses.

    python3 docs/gen_manual.py

The prose of the manual does NOT live here — it lives in
`docs/manual-markdown/manual.md`. See `docs/BUILDING-THE-MANUAL.md`.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INVENTORY = HERE / "inventory.json"
BUILD_DIR = HERE / "_build"

E = html.escape

# Every gotcha an area carries is rendered. Until this was fixed the render line ended
# `[:6]`: the eight areas hold 127 gotchas, 48 reached the page, and 79 of them — 62%,
# including one a correction pass had just rewritten — reached neither the HTML nor the
# PDF. Nothing in the build output said so, which is the part that mattered: a reader
# cannot ask about a warning they were never shown, and the owner had no way to notice
# that most of what they wrote was being thrown away. The limit below is a runaway guard
# against a pathological inventory, not an editorial cap — the longest list today is 23 —
# and `_report_gotchas` prints a line on every build, so if an area ever does reach it the
# build names the area and says how many were lost instead of quietly dropping them.
GOTCHA_LIMIT = 60

AREA_TITLE = {
    "incubation-studies": "Incubation studies and shared curation",
    "menus": "Menu bar",
    "file-and-calibration": "File Manager &amp; Calibration",
    "detection": "Preprocess &amp; Detection",
    "stats-merge-measure": "Fits, Merge &amp; Measure",
    "event-table-and-inspector": "Event table &amp; Event inspector",
    "event-gallery": "Event gallery",
    "plots": "Distributions, Figure Library &amp; publication export",
    "noise-iv-vdep-eventplots": "Noise, I-V, voltage dependence &amp; event waveforms",
}

# Screenshots shown at the head of each reference area: area key -> [(image stem, caption)].
# Image stems are file names (without extension) in docs/manual-markdown/images/.
AREA_IMG = {
    "detection": [
        ("panel_detection",
         "The <b>Detection</b> tab. The polarity line is measured, not assumed &mdash; here it reports "
         "<i>enhancement (down): 1969 down / 0 up (confidence 1.00, measured)</i>. The <code>range</code> "
         "row is read-only: the Schmitt detector always runs over the whole recording."),
        ("panel_ml",
         "The <b>ML detector</b> tab, with no model loaded &mdash; <code>Run ML detection</code> stays "
         "greyed until you <code>Load model&hellip;</code>. Its <b>Model preprocessing</b> block is a "
         "separate chain from the <code>Preprocess</code> tab: the preset must match the molecule the "
         "model was trained on, and the panel says so in amber."),
    ],
    "file-and-calibration": [
        ("inspector_dock",
         "The Inspector dock. <b>Gain</b>, <b>Offset</b> and <b>Target units</b> are text fields you "
         "type into &mdash; not drop-downs."),
    ],
    "stats-merge-measure": [
        ("panel_measure",
         "The <b>Manual measurement</b> tab. <b>Live statistics</b> read <code>&mdash;</code> until a "
         "cursor pair is placed; <code>Add to list</code> then stamps that row into the measurement "
         "table below, which the Plots window treats as another per-event table."),
    ],
    "event-gallery": [
        ("gallery_labels",
         "The Event gallery on the &minus;600&#8239;mV recording after merging, showing the "
         "<code>Label</code> / <code>Apply</code> / <code>Clear</code> toolbar and the "
         "<i>1&ndash;9 to label</i> hint."),
        ("gallery_real",
         "The Event gallery on the &minus;700&#8239;mV apo-ferritin recording &mdash; page 3 of 29, 984 "
         "events sorted by dwell. <b>Per page</b> is on <i>Fit</i>, which sized this page to 35 tiles "
         "for the window."),
    ],
    "event-table-and-inspector": [
        ("event_inspector",
         "The Event inspector with fitted CUSUM levels drawn as flat green segments."),
    ],
    "plots": [
        ("hist_dwell_fit",
         "The Plots window: histogram of dwell with a fitted curve, its goodness-of-fit verdict and a "
         "ranked table of rival models."),
    ],
    "noise-iv-vdep-eventplots": [
        ("psd", "The Power spectrum window."),
    ],
}


def load_inventory(path: Path = INVENTORY) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _figure(images: dict[str, str], stem: str, caption: str) -> str:
    """One <figure>, or "" when the image is not available."""
    if stem not in images:
        return ""
    if not stem.startswith('current_'):
        caption = '<b>Historical interface illustration.</b> ' + caption
    return (f'<figure class="shot"><img src="{images[stem]}" alt="" loading="lazy">'
            f"<figcaption>{caption}</figcaption></figure>")


def _report_gotchas(inv: list[dict]) -> None:
    """Say what happened to the gotcha lists, in the same style as gen_page.py's checks.

    `gen_page.py` prints one `  ok  …` line per self-check and one `! …` line when
    something it cannot fix has gone wrong; this is the reference section's own check,
    printed as `build_reference` runs, so it lands in the same block of build output.
    """
    counts = [(a.get("key", "?"), len(a.get("gotchas", []) or [])) for a in inv]
    total = sum(n for _, n in counts)
    longest = max((n for _, n in counts), default=0)
    lost = [(k, n - GOTCHA_LIMIT) for k, n in counts if n > GOTCHA_LIMIT]
    if not lost:
        print(f"  ok  {total} gotchas in {len(counts)} areas, all rendered "
              f"(longest list {longest}, limit {GOTCHA_LIMIT})")
        return
    print(f"! {sum(n for _, n in lost)} of {total} gotchas are NOT on the page — over the "
          f"{GOTCHA_LIMIT}-per-area limit in "
          + ", ".join(f"{k} (+{n})" for k, n in lost)
          + ". Raise GOTCHA_LIMIT in gen_manual.py; the manual is short until you do.")


def controls_table(controls: list[dict]) -> str:
    rows = []
    for c in controls:
        label = E(c.get("label", "") or "")
        kind = E((c.get("kind", "") or "").replace("-", " "))
        grp = E(c.get("group", "") or "")
        does = c.get("what_it_does", "") or ""
        notes = c.get("notes", "") or ""
        sc = c.get("shortcut", "") or ""
        corr = c.get("correction", "") or ""
        extra = []
        if sc:
            extra.append(f'<span class="kbd">{E(sc)}</span>')
        if notes:
            extra.append(f'<span class="note">{E(notes)}</span>')
        if corr:
            extra.append(f'<span class="corr"><b>Exactly:</b> {E(corr[:400])}</span>')
        rows.append(
            f'<tr><td class="c-label"><code>{label}</code>'
            + (f'<span class="grp">{grp}</span>' if grp else "")
            + f'</td><td class="c-kind">{kind}</td>'
            + f'<td class="c-what">{E(does)}' + "".join(extra) + "</td></tr>")
    return ('<div class="tablewrap"><table class="controls">'
            "<thead><tr><th>Label</th><th>Type</th><th>What it does</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table></div>")


def build_reference(images: dict[str, str] | None = None,
                    inventory: list[dict] | None = None,
                    skip_images: set[str] | None = None) -> tuple[str, int, list[tuple[str, str]]]:
    """Render the whole control reference.

    `images` maps an image stem to whatever belongs in an <img src="…"> — a data: URI
    when the page inlines its screenshots. `skip_images` names stems the prose already
    shows, so no screenshot appears twice on the page (and its bytes are embedded once).
    Returns (html, control count, [(anchor id, plain-text area title)]) so the caller can
    build a contents rail that cannot drift from the headings.
    """
    images = {k: v for k, v in (images or {}).items() if k not in (skip_images or set())}
    inv = inventory if inventory is not None else load_inventory()
    _report_gotchas(inv)
    out, toc, total = [], [], 0
    for area in inv:
        key = area["key"]
        title = AREA_TITLE.get(key, key)
        total += len(area["controls"])
        toc.append((f"ref-{key}", html.unescape(title).replace("‑", "-")))
        shots = "".join(_figure(images, stem, cap) for stem, cap in AREA_IMG.get(key, []))
        out.append(
            f'<section class="ref-area" id="ref-{key}">'
            f"<h3>{title} <span class='count'>{len(area['controls'])} controls</span></h3>"
            + (f'<p class="opened">{E(area["opened_by"])}</p>' if area.get("opened_by") else "")
            + f'<p class="lede">{E(area["summary"])}</p>'
            + shots
            + "".join(f'<p class="gotcha">{E(g)}</p>'
                      for g in (area.get("gotchas", []) or [])[:GOTCHA_LIMIT])
            + controls_table(area["controls"])
            + "</section>")
    return "".join(out), total, toc


def main() -> None:
    ref, total, toc = build_reference()
    BUILD_DIR.mkdir(exist_ok=True)
    out = BUILD_DIR / "reference.html"
    out.write_text(ref, encoding="utf-8")
    print(f"reference section built: {total} controls in {len(toc)} areas, "
          f"{len(ref) / 1024:.0f} KB -> {out.relative_to(HERE.parent)}")
    print("(screenshots are not inlined in this stand-alone copy; run gen_page.py for the real page)")


if __name__ == "__main__":
    main()
