"""Build the NanoScope manual as ONE self-contained HTML page.

Everything this needs is in the repository, and every path is resolved relative to this
file — you can clone, run it, and get the page:

    /opt/anaconda3/bin/python3 docs/gen_page.py            # the two HTML copies
    /opt/anaconda3/bin/python3 docs/gen_page.py --pdf --zip  # …and the PDF and the zip

Sources
    docs/manual-markdown/manual.md      the prose — the single source of truth. Edit
                                        that file, re-run this, never the reverse.
    docs/manual-markdown/images/*.webp  the screenshots; inlined as data: URIs, so the
                                        built page loads nothing from the network.
    docs/inventory.json                 the control inventory. The `Control reference`
                                        section is rendered from it by gen_manual.py —
                                        the markdown holds only that section's preamble,
                                        because the inventory carries per-control notes,
                                        shortcuts and corrections that a markdown table
                                        flattens away.

Outputs (all under docs/)
    manual.html and manual-markdown/manual.html   byte-identical copies
    nanoscope-manual.pdf   --pdf, needs Google Chrome; prints the page's own print CSS
    nanoscope-manual.zip   --zip, the markdown + images + html + pdf bundle

The build verifies itself: every <img> must be inlined, every internal link must resolve
to an id on the page, and the contents rail is generated from the headings, so it cannot
drift from them. Failures raise.
"""

from __future__ import annotations

import argparse
import base64
import html
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

try:
    import markdown
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("This build needs Python-Markdown:  /opt/anaconda3/bin/python3 -m pip install markdown")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_manual  # noqa: E402  (same directory)

HERE = Path(__file__).resolve().parent
MD_DIR = HERE / "manual-markdown"
MD_FILE = MD_DIR / "manual.md"
IMG_DIR = MD_DIR / "images"
OUT_HTML = HERE / "manual.html"
OUT_HTML_COPY = MD_DIR / "manual.html"
OUT_PDF = HERE / "nanoscope-manual.pdf"
OUT_ZIP = HERE / "nanoscope-manual.zip"

MIME = {".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", ".gif": "image/gif", ".svg": "image/svg+xml"}

CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

CSS = """
:root{
  --ink:#12161c; --ink-2:#39424e; --ink-3:#68727f;
  --bg:#f6f8fa; --surface:#ffffff; --rule:#dde3ea; --rule-2:#eef2f6;
  --trace:#2b6fb0; --trace-soft:#e8f0f8;
  --span:#a8670c; --span-soft:#fdf3e3;
  --checked:#00795f; --checked-soft:#e6f4f0;
  --code-bg:#eef2f7; --code-ink:#16324a;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ink:#e6ebf1; --ink-2:#aab6c4; --ink-3:#7e8b9a;
    --bg:#0e1217; --surface:#151b23; --rule:#252e39; --rule-2:#1b222b;
    --trace:#6fb2f0; --trace-soft:#12202f;
    --span:#e3a44a; --span-soft:#2a2013;
    --checked:#4fc4a6; --checked-soft:#10241f;
    --code-bg:#1b222b; --code-ink:#bcd3e8;
  }
}
:root[data-theme="dark"]{
  --ink:#e6ebf1; --ink-2:#aab6c4; --ink-3:#7e8b9a;
  --bg:#0e1217; --surface:#151b23; --rule:#252e39; --rule-2:#1b222b;
  --trace:#6fb2f0; --trace-soft:#12202f;
  --span:#e3a44a; --span-soft:#2a2013;
  --checked:#4fc4a6; --checked-soft:#10241f;
  --code-bg:#1b222b; --code-ink:#bcd3e8;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:Charter,"Bitstream Charter","Iowan Old Style",Georgia,"Times New Roman",serif;
  font-size:17px; line-height:1.62; -webkit-font-smoothing:antialiased;
}
code,.kbd,.mono{font-family:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Cascadia Mono",monospace}
h1,h2,h3,h4,.eyebrow,.rail,th{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
h1,h2,h3{text-wrap:balance; line-height:1.15; margin:0}
a{color:var(--trace)}
.wrap{display:grid; grid-template-columns:minmax(0,1fr); max-width:1180px; margin:0 auto; padding:0 20px}
@media (min-width:1000px){ .wrap{grid-template-columns:232px minmax(0,1fr); gap:44px} }

/* masthead */
header.top{border-bottom:1px solid var(--rule); background:var(--surface)}
.masthead{max-width:1180px;margin:0 auto;padding:44px 20px 34px}
.eyebrow{font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-3);margin:0 0 12px}
h1{font-size:clamp(30px,4.6vw,46px);font-weight:660;letter-spacing:-.022em}
.sub{margin:14px 0 0;max-width:62ch;color:var(--ink-2);font-size:19px}
.facts{display:flex;flex-wrap:wrap;gap:8px;margin-top:22px}
.fact{font-family:ui-monospace,monospace;font-size:12px;padding:5px 10px;border:1px solid var(--rule);
      border-radius:3px;color:var(--ink-2);background:var(--bg)}

/* rail */
nav.rail{font-size:13.5px}
@media (min-width:1000px){ nav.rail{position:sticky;top:0;align-self:start;max-height:100vh;overflow:auto;padding:34px 0 40px} }
@media (max-width:999px){ nav.rail{padding:22px 0 4px;border-bottom:1px solid var(--rule);margin-bottom:8px} }
nav.rail .grp{font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--ink-3);margin:20px 0 7px}
nav.rail a{display:block;padding:3px 0;color:var(--ink-2);text-decoration:none;border-left:2px solid transparent;padding-left:11px;margin-left:-11px}
nav.rail a:hover{color:var(--trace);border-left-color:var(--trace)}

main{padding:34px 0 90px;min-width:0}
section{margin:0 0 54px; scroll-margin-top:20px}
h2{font-size:27px;font-weight:640;letter-spacing:-.016em;padding-bottom:9px;border-bottom:2px solid var(--ink);margin-bottom:20px}
h3{font-size:20px;font-weight:620;margin:34px 0 12px;letter-spacing:-.01em}
h4{font-size:15.5px;font-weight:640;margin:24px 0 8px;color:var(--ink-2)}
p{margin:0 0 15px;max-width:68ch}
ul,ol{max-width:68ch;padding-left:22px;margin:0 0 15px}
li{margin:0 0 7px}
strong{font-weight:640}
code{background:var(--code-bg);color:var(--code-ink);padding:1.5px 5px;border-radius:3px;font-size:.855em}

/* figures */
.shot{margin:22px 0 26px}
.shot img{width:100%;height:auto;display:block;border:1px solid var(--rule);border-radius:5px;background:var(--surface)}
.shot figcaption{font-size:13.5px;color:var(--ink-3);margin-top:9px;max-width:70ch;line-height:1.5}
.shot.narrow img{max-width:620px}
.shot.half img{max-width:900px}

/* callouts */
.callout{border-left:3px solid var(--trace);background:var(--trace-soft);padding:13px 17px;margin:20px 0;border-radius:0 4px 4px 0}
.callout.warn{border-left-color:var(--span);background:var(--span-soft)}
.callout.good{border-left-color:var(--checked);background:var(--checked-soft)}
.callout p{margin:0 0 8px} .callout p:last-child{margin:0}
.callout .t{font-family:system-ui,sans-serif;font-size:11px;letter-spacing:.11em;text-transform:uppercase;color:var(--ink-3);display:block;margin-bottom:5px}

/* steps */
ol.steps{list-style:none;padding:0;counter-reset:s;max-width:70ch}
ol.steps>li{counter-increment:s;position:relative;padding-left:40px;margin-bottom:18px}
ol.steps>li::before{content:counter(s);position:absolute;left:0;top:1px;width:26px;height:26px;border-radius:50%;
  background:var(--ink);color:var(--bg);font-family:ui-monospace,monospace;font-size:12.5px;display:grid;place-items:center}

/* data tables */
.tablewrap{overflow-x:auto;margin:16px 0 8px;border:1px solid var(--rule);border-radius:5px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14px;font-family:system-ui,sans-serif}
th{text-align:left;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3);
   padding:9px 13px;border-bottom:1px solid var(--rule);background:var(--bg);position:sticky;top:0}
td{padding:9px 13px;border-bottom:1px solid var(--rule-2);vertical-align:top}
tr:last-child td{border-bottom:none}
table.num td{font-variant-numeric:tabular-nums}
.controls .c-label{width:22%;min-width:170px}
.controls .c-label code{background:none;padding:0;color:var(--ink);font-weight:600;font-size:13px}
.controls .c-kind{width:11%;min-width:88px;color:var(--ink-3);font-size:12.5px;white-space:nowrap}
.controls .c-what{color:var(--ink-2);line-height:1.5}
.grp{display:block;font-size:11px;color:var(--ink-3);margin-top:3px;font-family:system-ui,sans-serif}
.note{display:block;margin-top:6px;font-size:12.5px;color:var(--ink-3);padding-left:10px;border-left:2px solid var(--rule)}
.corr{display:block;margin-top:6px;font-size:12.5px;color:var(--span);padding-left:10px;border-left:2px solid var(--span)}
.kbd{display:inline-block;margin-top:6px;margin-right:6px;font-size:11.5px;padding:2px 6px;border:1px solid var(--rule);border-radius:3px;color:var(--ink-2)}
.count{font-family:ui-monospace,monospace;font-size:12px;font-weight:400;color:var(--ink-3);margin-left:8px}
.opened{font-size:13px;color:var(--ink-3);margin:0 0 10px;max-width:70ch}
.lede{color:var(--ink-2)}
.readout{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--ink-3);
  border-radius:0 4px 4px 0;padding:12px 15px;margin:14px 0 18px;overflow-x:auto;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;line-height:1.55;
  color:var(--ink-2);white-space:pre}
.readout b{color:var(--ink);font-weight:600}
.rcap{font-family:system-ui,sans-serif;font-size:11px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);margin:18px 0 -8px}
.settings{margin:14px 0 18px}
.settings table{font-size:13.5px}
.settings td:first-child{font-family:ui-monospace,monospace;font-size:12.5px;width:34%;color:var(--ink)}
.check{border-left:3px solid var(--checked);background:var(--checked-soft);padding:11px 15px;margin:14px 0;
  border-radius:0 4px 4px 0;font-size:14.5px}
.check b{font-family:system-ui,sans-serif;font-size:11px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);display:block;margin-bottom:4px;font-weight:600}
.gotcha{font-size:14px;color:var(--span);padding-left:12px;border-left:2px solid var(--span);max-width:68ch}
.ref-area{margin-bottom:44px}
footer{border-top:1px solid var(--rule);padding:26px 0 60px;color:var(--ink-3);font-size:13.5px}
:focus-visible{outline:2px solid var(--trace);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}

/* ---- print / save-as-PDF -------------------------------------------------------------
   The page is one self-contained file, so "download it, open it, Cmd/Ctrl+P, Save as PDF"
   is the route to a paper copy — and the browser's own renderer is better than anything
   this project could ship. These rules make that output a document rather than a
   screenshot of a website. */
@media print{
  :root{
    --ink:#000; --ink-2:#222; --ink-3:#555; --bg:#fff; --surface:#fff;
    --rule:#bbb; --rule-2:#ddd; --code-bg:#f2f2f2; --code-ink:#000;
    --trace:#14538c; --trace-soft:#f2f6fa; --span:#8a5200; --span-soft:#fbf6ee;
    --checked:#005c46; --checked-soft:#f0f6f4;
  }
  @page{ margin:16mm 14mm; }
  body{font-size:10.5pt; line-height:1.5; background:#fff}
  nav.rail{display:none}
  .wrap{display:block; max-width:none; padding:0}
  header.top{border:none}
  .masthead{padding:0 0 18pt}
  h1{font-size:26pt}
  .facts{gap:5pt}
  .fact{font-size:8pt}
  main{padding:0}
  section{margin-bottom:20pt; break-inside:auto}
  h2{font-size:15pt; break-after:avoid; break-before:page; padding-bottom:5pt}
  /* the masthead prints as a title page of its own, so every section — including
     the first — may start on a fresh page */
  h3,h4{break-after:avoid}
  p,li{orphans:3; widows:3}
  .shot{break-inside:avoid; margin:12pt 0}
  .shot img{max-width:100%; border-color:#ccc}
  .shot figcaption{font-size:8.5pt}
  .callout,.check,.readout,.settings{break-inside:avoid}
  .readout{font-size:8pt; white-space:pre-wrap}
  .tablewrap{overflow:visible; break-inside:auto}
  table{font-size:8.5pt}
  thead{display:table-header-group}
  tr{break-inside:avoid}
  th{position:static; background:#f4f4f4}
  .controls .c-label code{font-size:8.5pt}
  .note,.corr,.grp{font-size:8pt}
  footer{break-before:page}
}

/* block labels: a paragraph in the markdown that is nothing but bold text */
.lbl{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-size:11.5px;
  letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);font-weight:640;
  margin:26px 0 7px;break-after:avoid}
.lbl+.readout,.lbl+.tablewrap,.lbl+.shot{margin-top:6px}
nav.rail a.sub{font-size:12.5px;color:var(--ink-3);padding-left:22px}
.prov{font-style:italic}
hr{border:none;border-top:1px solid var(--rule);margin:34px 0}
@media print{ .lbl{font-size:8.5pt;margin:12pt 0 4pt} }
"""


# ----------------------------------------------------------------------------- images
def load_images(img_dir: Path = IMG_DIR) -> dict[str, str]:
    """Every screenshot in `img_dir`, as {file stem: data: URI}."""
    out = {}
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in MIME and p.is_file():
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            out[p.stem] = f"data:{MIME[p.suffix.lower()]};base64,{b64}"
    if not out:
        raise SystemExit(f"no images found in {img_dir}")
    return out


# ------------------------------------------------------------------------ front matter
def split_front_matter(text: str) -> tuple[str, str]:
    """Everything before the first `## ` heading is masthead material, not body."""
    m = re.search(r"^## ", text, re.M)
    if not m:
        raise SystemExit("manual.md has no '## ' heading — cannot find where the body starts")
    return text[:m.start()], text[m.start():]


def _inline_md(text: str) -> str:
    """Render a fragment of markdown, without the wrapping <p>."""
    out = markdown.markdown(text.strip())
    return re.sub(r"^<p>|</p>$", "", out).strip()


def parse_front_matter(fm: str) -> dict:
    """Pull the masthead out of the markdown's own opening, with tolerant fallbacks."""
    lines = fm.splitlines()
    h1s = [i for i, ln in enumerate(lines) if ln.startswith("# ")]
    if not h1s:
        raise SystemExit("manual.md has no '# ' title")
    doc_title = lines[h1s[0]][2:].strip()   # "NanoScope — User Manual": the browser tab
    h1 = h1s[-1]                            # the last H1 is the masthead itself
    title = lines[h1][2:].strip()

    eyebrow = next((ln.strip() for ln in lines[:h1]
                    if ln.strip() and not ln.lstrip().startswith(("#", ">", "*", "-", "|", "_"))),
                   "User manual")
    provenance = next((ln.strip().strip("*") for ln in lines[:h1]
                       if ln.strip().startswith("*") and not ln.strip().startswith("* ")), "")

    sub, facts, i = [], [], h1 + 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    while i < len(lines) and lines[i].strip() and not lines[i].startswith(("-", "#", ">")):
        sub.append(lines[i].strip())
        i += 1
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("- "):
            facts.append(ln[2:].strip())
        elif ln and not ln.startswith("-"):
            break
        i += 1
    return {"title": title, "doc_title": doc_title, "eyebrow": eyebrow, "provenance": provenance,
            "sub": " ".join(sub), "facts": facts}


# --------------------------------------------------------------------- markdown → html
IMG_TAG = re.compile(r'<p>\s*(<img\b[^>]*?>)\s*</p>\s*(?:<p><em>(.*?)</em></p>)?', re.S)
ATTR = re.compile(r'(\w+)\s*=\s*"(.*?)"', re.S)


def _figures(body: str, images: dict[str, str]) -> str:
    """`![alt](images/x.webp)` + its italic caption line -> one <figure class="shot">.

    Each figure keeps a `data-img` attribute naming the file it came from: it survives
    into the page, so a reader can tell which screenshot they are looking at, and the
    build uses it to see which shots the prose already spends.
    """
    missing: list[str] = []

    def repl(m: re.Match) -> str:
        attrs = dict(ATTR.findall(m.group(1)))
        src = attrs.get("src", "")
        stem = Path(src).stem
        if stem not in images:
            missing.append(src)
            return m.group(0)
        caption = (m.group(2) or attrs.get("alt", "")).strip()
        return (f'<figure class="shot" data-img="{stem}">'
                f'<img src="{images[stem]}" alt="" loading="lazy">'
                f"<figcaption>{caption}</figcaption></figure>")

    out = IMG_TAG.sub(repl, body)
    if missing:
        raise SystemExit("manual.md references images that are not in %s: %s"
                         % (IMG_DIR, ", ".join(sorted(set(missing)))))
    return out


def _readouts(body: str) -> str:
    """Fenced blocks are quoted app output — give them the readout styling."""
    return re.sub(r"<pre><code[^>]*>(.*?)</code></pre>",
                  lambda m: '<div class="readout">%s</div>' % m.group(1).rstrip("\n"),
                  body, flags=re.S)


def _labels(body: str) -> str:
    """A paragraph that is nothing but bold text is a block label, not a sentence."""
    return re.sub(r"<p><strong>(.*?)</strong></p>",
                  lambda m: '<p class="lbl">%s</p>' % m.group(1), body, flags=re.S)


def _quotes(body: str) -> str:
    return re.sub(r"<blockquote>\s*(.*?)\s*</blockquote>",
                  lambda m: '<div class="callout">%s</div>' % m.group(1), body, flags=re.S)


def _tables(body: str) -> str:
    """Tables scroll inside their own box rather than pushing the page sideways."""
    return re.sub(r"<table>(.*?)</table>",
                  lambda m: '<div class="tablewrap"><table>%s</table></div>' % m.group(1),
                  body, flags=re.S)


H2 = re.compile(r'<h2 id="([^"]+)">(.*?)</h2>', re.S)
REFERENCE_SLOT = "<!--CONTROL-REFERENCE-->"


def _sectionise(body: str) -> tuple[str, list[tuple[str, str]]]:
    """Wrap each H2 and its content in a <section id>, and swap in the control reference.

    The markdown's `Control reference` section keeps its own preamble; everything from
    its first H3 onwards is dropped and REFERENCE_SLOT put in its place, for the caller
    to fill with the inventory-rendered tables.
    """
    parts = H2.split(body)
    lead = parts[0].strip()
    out, toc = ([lead] if lead else []), []
    for i in range(1, len(parts), 3):
        anchor, title_html, content = parts[i], parts[i + 1], parts[i + 2]
        title = re.sub(r"<[^>]+>", "", title_html).strip()
        if re.match(r"control reference", title, re.I):
            cut = content.find("<h3")
            content = (content[:cut] if cut >= 0 else content) + REFERENCE_SLOT
        out.append(f'<section id="{anchor}"><h2>{title_html}</h2>{content}</section>')
        toc.append((anchor, title))
    return "".join(out), toc


def render_body(md_text: str,
                images: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "toc"],
                           extension_configs={"toc": {"permalink": False}})
    body = md.convert(md_text)
    body = _figures(body, images)
    body = _readouts(body)
    body = _labels(body)
    body = _quotes(body)
    body = _tables(body)
    return _sectionise(body)


# ------------------------------------------------------------------------ contents rail
RAIL_GROUPS = [(re.compile(r"^walkthrough", re.I), "Walkthroughs"),
               (re.compile(r"^control reference", re.I), "Reference"),
               (re.compile(r"^what the app refuses", re.I), "End matter")]


def rail_label(title: str) -> str:
    t = re.sub(r"^Walkthrough\s+(\d+)\s*[—–-]\s*", lambda m: m.group(1) + " · ", title)
    for sep in (":", " — ", " – "):
        if sep in t and len(t) > 26:
            t = t.split(sep)[0].strip()
    if len(t) > 30:
        t = t[:30].rsplit(" ", 1)[0] + "…"
    return t


def build_rail(toc: list[tuple[str, str]], ref_toc: list[tuple[str, str]]) -> str:
    rows, group = ['<nav class="rail">', '<div class="grp">Start</div>'], "Start"
    for anchor, title in toc:
        for pat, name in RAIL_GROUPS:
            if pat.match(title) and name != group:
                rows.append(f'<div class="grp">{name}</div>')
                group = name
                break
        rows.append(f'<a href="#{anchor}">{html.escape(rail_label(title))}</a>')
        if re.match(r"control reference", title, re.I):
            rows += [f'<a class="sub" href="#{a}">{html.escape(t)}</a>' for a, t in ref_toc]
    rows.append("</nav>")
    return "\n".join(rows)


# ------------------------------------------------------------------------------- page
def build_page() -> str:
    images = load_images()
    text = MD_FILE.read_text(encoding="utf-8")
    front, body_md = split_front_matter(text)
    meta = parse_front_matter(front)
    body, toc = render_body(body_md, images)
    # Only the figures that survived into the page count as "already shown" — the ones
    # inside the replaced control-reference subsections did not.
    in_prose = set(re.findall(r'data-img="([^"]+)"', body))
    reference, total, ref_toc = gen_manual.build_reference(images, skip_images=in_prose)
    if REFERENCE_SLOT not in body:
        raise SystemExit("manual.md has no 'Control reference' heading to hang the "
                         "inventory tables on")
    body = body.replace(REFERENCE_SLOT, reference)
    rail = build_rail(toc, ref_toc)

    facts = "".join(f'<span class="fact">{_inline_md(f)}</span>' for f in meta["facts"])
    prov = f'<p class="prov">{_inline_md(meta["provenance"])}</p>' if meta["provenance"] else ""
    head = (f'<!doctype html>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f'<title>{html.escape(meta["doc_title"])}</title>\n<style>{CSS}</style>\n')
    return head + f"""<header class="top"><div class="masthead">
  <p class="eyebrow">{_inline_md(meta["eyebrow"])}</p>
  <h1>{html.escape(meta["title"])}</h1>
  <p class="sub">{_inline_md(meta["sub"])}</p>
  <div class="facts">{facts}</div>
</div></header>

<div class="wrap">{rail}<main>
{body}
<footer>
<p>NanoScope 0.1.0 &middot; This manual documents {total} controls across {len(ref_toc)} areas.</p>
{prov}
<p>Built from <code>docs/manual-markdown/manual.md</code> and <code>docs/inventory.json</code> by
<code>docs/gen_page.py</code>. {len(images)} screenshots are embedded in this file; it loads nothing
from the network.</p>
</footer>
</main></div>
"""


# ------------------------------------------------------------------------------ verify
def verify(page: str) -> list[str]:
    """Everything the page promises about itself, checked. Returns a list of notes."""
    notes, problems = [], []

    ids = set(re.findall(r'\bid="([^"]+)"', page))
    hrefs = [h for h in re.findall(r'href="#([^"]+)"', page)]
    dangling = sorted({h for h in hrefs if h not in ids})
    if dangling:
        problems.append("internal links with no target: " + ", ".join(dangling))
    notes.append(f"{len(hrefs)} internal links, all resolve to ids on the page")

    srcs = re.findall(r'<img\b[^>]*?src="([^"]{0,40})', page)
    external = [s for s in srcs if not s.startswith("data:")]
    if external:
        problems.append("images not inlined: " + ", ".join(external))
    notes.append(f"{len(srcs)} <img> tags, all data: URIs")

    figs, caps = page.count("<figure"), page.count("<figcaption")
    if figs != caps:
        problems.append(f"{figs} figures but {caps} captions")

    heads = re.findall(r'<section id="([^"]+)"><h2>', page)
    rail = re.findall(r'<nav class="rail">(.*?)</nav>', page, re.S)
    rail_targets = re.findall(r'href="#([^"]+)"', rail[0]) if rail else []
    missing = [h for h in heads if h not in rail_targets]
    if missing:
        problems.append("headings missing from the contents rail: " + ", ".join(missing))
    notes.append(f"{len(heads)} top-level sections, all in the contents rail "
                 f"({len(rail_targets)} rail entries)")

    for tag in ("script", "iframe", "link rel"):
        if f"<{tag}" in page:
            problems.append(f"page is not self-contained: it has a <{tag}>")

    if problems:
        raise SystemExit("BUILD FAILED\n  - " + "\n  - ".join(problems))
    return notes


# -------------------------------------------------------------------------- pdf / zip
def build_pdf(html_path: Path, pdf_path: Path, wait_s: float = 420) -> bool:
    """Print the page with Chrome — the same renderer the manual tells you to print with.

    Chrome 151's headless mode writes the PDF and then does not exit, so this watches the
    file instead of the process: once its size has stopped changing, the print is done and
    the browser is stopped.
    """
    if not CHROME.exists():
        print(f"! no Chrome at {CHROME} — PDF not rebuilt, the existing one is stale")
        return False
    pdf_path.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as profile:
        cmd = [str(CHROME), "--headless=new", "--disable-gpu", "--no-first-run",
               f"--user-data-dir={profile}", "--no-pdf-header-footer",
               "--virtual-time-budget=120000",
               f"--print-to-pdf={pdf_path}", html_path.as_uri()]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            last, stable, t0 = -1, 0, time.time()
            while time.time() - t0 < wait_s and proc.poll() is None:
                size = pdf_path.stat().st_size if pdf_path.exists() else 0
                stable = stable + 1 if size and size == last else 0
                last = size
                if stable >= 3:
                    break
                time.sleep(1.0)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    proc.kill()
    if not pdf_path.exists() or pdf_path.stat().st_size < 10_000:
        print("! Chrome did not produce a PDF")
        return False
    return True


def build_zip(zip_path: Path) -> None:
    root = "nanoscope-manual-md"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(MD_FILE, f"{root}/manual.md")
        z.write(OUT_HTML_COPY, f"{root}/manual.html")
        if OUT_PDF.exists():
            z.write(OUT_PDF, f"{root}/nanoscope-manual.pdf")
        for p in sorted(IMG_DIR.iterdir()):
            if p.suffix.lower() in MIME and p.is_file():
                z.write(p, f"{root}/images/{p.name}")


# -------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdf", action="store_true", help="also print a PDF with headless Chrome")
    ap.add_argument("--zip", action="store_true", help="also rebuild the distribution zip")
    args = ap.parse_args()

    page = build_page()
    for note in verify(page):
        print("  ok  " + note)

    OUT_HTML.write_text(page, encoding="utf-8")
    shutil.copyfile(OUT_HTML, OUT_HTML_COPY)
    mb = OUT_HTML.stat().st_size / 1e6
    print(f"wrote {OUT_HTML.relative_to(HERE.parent)} and "
          f"{OUT_HTML_COPY.relative_to(HERE.parent)}  ({mb:.2f} MB, identical copies)")

    if args.pdf and build_pdf(OUT_HTML, OUT_PDF):
        print(f"wrote {OUT_PDF.relative_to(HERE.parent)}  "
              f"({OUT_PDF.stat().st_size / 1e6:.2f} MB)")
    if args.zip:
        build_zip(OUT_ZIP)
        print(f"wrote {OUT_ZIP.relative_to(HERE.parent)}  "
              f"({OUT_ZIP.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
