"""Paper-style article of the project (M17, docs/tasks/M17-article.md): generated blocks.

    python -m tools.article all        # collect then render (make article)
    python -m tools.article collect    # sources → docs/article/chiffres.json
    python -m tools.article render     # chiffres.json + figure registry → article.md blocks
    python -m tools.article --check    # make ci: blocks up to date, keys, figures, numbers
    python -m tools.article pdf        # all, then pandoc → build/article/ (make article-pdf)

A block sits between two HTML comments, invisible once rendered:

    <!-- article:n:map.voc.int8 -->55.66<!-- /article -->
    <!-- article:table:map.stades --> … <!-- /article -->
    <!-- article:fig:graphe/graphe_tiny-yolov2-voc --> … <!-- /article -->

Its content is rewritten by `render` and never edited by hand; everything outside blocks
(the narrative) is left byte for byte. `collect` reads each key of the `CLES` registry from
its source (JSON of `build/`, Markdown or CSV of `results/`); when a source is missing the
value already in `chiffres.json` is kept and reported, so the article renders anywhere
(CI, Colab, a machine without `build/`). Standard library only (ADR 0001); the figure
registry `tools.figures` is imported lazily and never loads matplotlib.
"""

import argparse
import csv
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT / "docs" / "article" / "article.md"
DATA = ROOT / "docs" / "article" / "chiffres.json"
SUIVI = ROOT / "docs" / "tasks" / "README.md"
BUILD = ROOT / "build" / "article"

INLINE = ("n", "rev")  # rendered on one line; `etat:resume` too
TYPES = ("n", "table", "fig", "etat", "rev", "list")

# Status of a number (M17, "Statut de chaque chiffre"); `measured` carries no mark.
STATUSES = {"measured": "", "csim": "C-sim", "projection": "proj.", "estimate": "est.",
            "tier-R": "tier R", "published": "pub."}
PENDING = ("projection", "estimate", "tier-R", "csim")  # listed in §11 (list:status)

# Sections whose narrative may not quote a decimal number outside a block, and the
# constants of the network or of the protocol that may appear there.
RESULT_SECTIONS = re.compile(r"^## (Abstract|(4|5|6|7|8|9|10)\.)")
WHITELIST = {"0.1", "0.45", "0.005", "0.5", "0.25", "0.3", "1.0"}

OPEN = re.compile(r"<!-- article:([a-z]+)(?::(\S+?))? -->")
CLOSE = re.compile(r"<!-- /article -->")


class ArticleError(Exception):
    """Malformed article (block not closed, nested, unknown type…)."""


class MissingSource(Exception):
    """Source of a key absent: the value of chiffres.json is kept."""


# --------------------------------------------------------------------------- blocks

@dataclass
class Block:
    type: str
    key: str
    start: int      # offset of the opening comment
    body: int       # offset of the content
    end: int        # offset of the closing comment
    line: int       # line of the opening comment (1-based)

    @property
    def content(self):
        return self._text[self.body:self.end]

    @property
    def tag(self):
        return f"{self.type}:{self.key}" if self.key else self.type


def _line(text, pos):
    return text.count("\n", 0, pos) + 1


def parse(text):
    """Blocks of `text`, in order. Unknown type, nested, unclosed or orphan close → error."""
    tokens = sorted([(m.start(), "open", m) for m in OPEN.finditer(text)]
                    + [(m.start(), "close", m) for m in CLOSE.finditer(text)],
                    key=lambda t: t[0])
    blocks, cur = [], None
    for pos, kind, m in tokens:
        if kind == "open":
            if cur is not None:
                raise ArticleError(f"line {_line(text, pos)}: block opened inside the block "
                                   f"of line {_line(text, cur.start())}")
            if m.group(1) not in TYPES:
                raise ArticleError(f"line {_line(text, pos)}: unknown block type "
                                   f"'{m.group(1)}' (known: {', '.join(TYPES)})")
            cur = m
        else:
            if cur is None:
                raise ArticleError(f"line {_line(text, pos)}: <!-- /article --> without an "
                                   "opening block")
            b = Block(cur.group(1), cur.group(2) or "", cur.start(), cur.end(), pos,
                      _line(text, cur.start()))
            b._text = text
            blocks.append(b)
            cur = None
    if cur is not None:
        raise ArticleError(f"line {_line(text, cur.start())}: block "
                           f"'{cur.group(1)}:{cur.group(2) or ''}' is never closed")
    return blocks


# ----------------------------------------------------------------------- formatting

def fmt(v, nd=None, sign=False):
    """English number format: decimal point, comma thousands separator, minus sign −."""
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if nd is None:
        nd = 0 if isinstance(v, int) else 2
    s = f"{abs(v):,.{nd}f}"
    if v < 0 and float(s.replace(",", "")) != 0:
        return "−" + s
    return ("+" + s) if sign and v > 0 else s


def mark(status):
    m = STATUSES.get(status, status)
    return f"<sup>{m}</sup>" if m else ""


def render_n(entry):
    return fmt(entry["value"], entry.get("nd"), entry.get("sign", False)) + mark(entry["status"])


def render_table(entry):
    v = entry["value"]
    nds = v.get("nd") or [None] * len(v["header"])
    signs = v.get("sign") or [False] * len(v["header"])
    lines = ["| " + " | ".join(v["header"]) + " |", "|" + "---|" * len(v["header"])]
    for row in v["rows"]:
        cells = [fmt(c, nd, s) if not isinstance(c, str) else c
                 for c, nd, s in zip(row, nds, signs)]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    if "status" in v:
        lines += ["", f"*Status of every number of this table: {v['status']}.*"]
    return "\n".join(lines)


def figure_of(key, figures, root=ROOT):
    """(Figure, png path) of `fig:<name>` or `fig:<name>/<file>`; error if unpublished."""
    name, _, stem = key.partition("/")
    if name not in figures:
        raise ArticleError(f"figure '{name}' absent from the registry (tools/figures)")
    f = figures[name]
    if f.subset or f.dest is not None:
        raise ArticleError(f"figure '{name}' is not published in results/figures/ "
                           "(subset of images, M12 rule)")
    png = Path(root) / "results" / "figures" / f.family / f"{stem or name}.png"
    if not png.exists():
        raise ArticleError(f"figure '{key}': {png.relative_to(root)} absent "
                           f"(regenerate: {f.command})")
    return f, png


def render_fig(key, figures, number, article, root=ROOT):
    f, png = figure_of(key, figures, root)
    rel = os.path.relpath(png, Path(article).parent).replace(os.sep, "/")
    return (f"![{png.stem}]({rel})\n\n"
            f"*Figure {number} (`{f.name}`): {f.caption}* <!-- {f.command} -->")


def suivi(path=SUIVI):
    """Rows (milestone, tasks, done) of the **Suivi** table of docs/tasks/README.md."""
    rows, inside = [], False
    for line in Path(path).read_text().splitlines():
        if line.startswith("## "):
            inside = line.strip() == "## Suivi"
            continue
        m = re.match(r"\|\s*(M[\d.]+)\s*\|\s*(\d+)[^|]*\|\s*(\d+)", line) if inside else None
        if m:
            rows.append((m.group(1), int(m.group(2)), int(m.group(3))))
    if not rows:
        raise ArticleError(f"no Suivi table in {path}")
    return rows


def render_etat(key, path=SUIVI):
    rows = suivi(path)
    total, done = sum(r[1] for r in rows), sum(r[2] for r in rows)
    full = sum(r[1] == r[2] for r in rows)
    if key == "resume":
        return (f"{done} of {total} tasks done across {len(rows)} milestones, "
                f"{full} milestones complete")
    if key:
        raise ArticleError(f"unknown variant etat:{key} (etat or etat:resume)")
    lines = ["| Milestone | Tasks | Done | Progress |", "|---|---|---|---|"]
    lines += [f"| {m} | {t} | {d} | {round(100 * d / t)} % |" for m, t, d in rows]
    lines.append(f"| **Total** | **{total}** | **{done}** | **{round(100 * done / total)} %** |")
    return "\n".join(lines)


def render_list(key, data):
    if key != "status":
        raise ArticleError(f"unknown variant list:{key} (list:status)")
    lines = []
    for st in PENDING:
        keys = [k for k, e in sorted(data.items()) if not k.startswith("_")
                and e.get("status") == st]
        if keys:
            lines.append(f"- **{st}** ({len(keys)}): " + ", ".join(f"`{k}`" for k in keys))
    return "\n".join(lines) or "- none"


def render_rev(data):
    r = data.get("_rev")
    return f"rev. `{r['rev']}`, {r['date']}" if r else "rev. —"


def _figures():
    sys.path.insert(0, str(ROOT))
    from tools.figures import FIGURES

    return FIGURES


def render(text, data, figures=None, article=ARTICLE, root=ROOT, suivi_path=SUIVI):
    """`text` with every block rewritten from `data`; the rest is left byte for byte."""
    figures = _figures() if figures is None else figures
    out, pos, n_fig = [], 0, 0
    for b in parse(text):
        if b.type == "n":
            content = render_n(_entry(data, b))
        elif b.type == "table":
            content = render_table(_entry(data, b))
        elif b.type == "fig":
            n_fig += 1
            content = render_fig(b.key, figures, n_fig, article, root)
        elif b.type == "etat":
            content = render_etat(b.key, suivi_path)
        elif b.type == "list":
            content = render_list(b.key, data)
        else:
            content = render_rev(data)
        inline = b.type in INLINE or (b.type == "etat" and b.key == "resume")
        out += [text[pos:b.body], content if inline else f"\n{content}\n"]
        pos = b.end
    out.append(text[pos:])
    return "".join(out)


def _entry(data, b):
    if b.key not in data:
        raise ArticleError(f"line {b.line}: key '{b.key}' absent from chiffres.json "
                           "(add it to CLES, then make article)")
    return data[b.key]


# --------------------------------------------------------------------------- checks

_STRIP = [re.compile(p, re.DOTALL) for p in (
    r"<!--.*?-->", r"`[^`]*`", r"\$[^$]*\$", r"\]\([^)]*\)", r"\[[^\]]*#[^\]]*\]",
    r"https?://\S+")]
_DECIMAL = re.compile(r"(?<![\w#§.\-/])\d+\.\d+(?![\w.])")


def stray_numbers(text):
    """[(line, number)] of decimal numbers outside blocks in the result sections."""
    masked = list(text)
    for b in parse(text):
        for i in range(b.body, b.end):
            if masked[i] != "\n":
                masked[i] = " "
    found, checked = [], False
    lines = "".join(masked).split("\n")
    for i, line in enumerate(lines, 1):
        if line.startswith("## "):
            checked = bool(RESULT_SECTIONS.match(line))
            continue
        if not checked:
            continue
        for rx in _STRIP:
            line = rx.sub(" ", line)
        found += [(i, m.group()) for m in _DECIMAL.finditer(line) if m.group() not in WHITELIST]
    return found


def check(text, data, figures=None, article=ARTICLE, root=ROOT, suivi_path=SUIVI):
    """Problems as strings (empty list: the article is up to date)."""
    try:
        blocks = parse(text)
    except ArticleError as e:
        return [str(e)]
    figures = _figures() if figures is None else figures
    problems, n_fig = [], 0
    for b in blocks:
        try:
            if b.type == "fig":
                n_fig += 1
            single = text[b.start:b.end + len("<!-- /article -->")]
            want = render(single, data, figures, article, root, suivi_path)
            if b.type == "fig":  # numbering depends on the position in the article
                want = re.sub(r"\*Figure 1 ", f"*Figure {n_fig} ", want, count=1)
            if want != single:
                found = b.content.strip().splitlines() or [""]
                exp = want[b.body - b.start:].removesuffix("<!-- /article -->").strip()
                problems.append(f"line {b.line}: {b.tag}: block differs from render; "
                                f"expected '{(exp.splitlines() or [''])[0][:70]}', "
                                f"found '{found[0][:70]}'")
        except ArticleError as e:
            problems.append(f"line {b.line}: {b.tag}: {e}")
    problems += [f"line {i}: number {n} outside a block in a result section"
                 for i, n in stray_numbers(text)]
    return problems


# ------------------------------------------------------------------------ git, data

def git_rev(root=ROOT):
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, check=True,
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=root, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return rev + ("-dirty" if dirty.strip() else "")


def today():
    return datetime.datetime.now().astimezone().date().isoformat()


def load(path=DATA):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}


def save(data, path=DATA):
    Path(path).write_text(json.dumps(dict(sorted(data.items())), indent=1,
                                     ensure_ascii=False) + "\n")


# --------------------------------------------------------------- source readers

def need(path):
    path = Path(path)
    if not path.exists():
        raise MissingSource(f"{_rel(path)} absent")
    return path


def _rel(p):
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def num(cell):
    """First number of a Markdown cell: `**56,30**`, `−0,64 point`, `4 952`, `26.87`."""
    s = cell.replace("**", "").replace("−", "-").replace(" ", " ").replace("\xa0", " ")
    m = re.search(r"-?\d+(?: \d{3})*(?:[.,]\d+)?", s)
    if not m:
        return None
    t = m.group().replace(" ", "").replace(",", ".")
    return float(t) if "." in t else int(t)


def md_tables(path):
    """[(heading, header, rows)] of the Markdown tables of `path` (cells stripped)."""
    out, heading, cur = [], "", None
    for line in need(path).read_text().splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cur is None:
                cur = (heading, cells, [])
                out.append(cur)
            elif not set("".join(cells)) <= set("-: "):
                cur[2].append(cells)
        else:
            cur = None
    return out


def table_with(path, *words):
    """First table whose header contains every word of `words`."""
    for heading, header, rows in md_tables(path):
        if all(any(w in h for h in header) for w in words):
            return header, rows
    raise MissingSource(f"{_rel(path)}: no table with columns {words}")


def row_with(rows, *words, col=0):
    for r in rows:
        if all(w in r[col] for w in words):
            return r
    raise MissingSource(f"no row with {words}")


def _results(name, root):
    return Path(root) / "results" / name


NET = "tiny-yolov2-voc"


def read_stages(root):
    """Stages of results/map_stades.md, or of build/m8/<net>/map_stades.json if present:
    {key: (map, equal images, compared images)} with key flottant | entier | csim | carte."""
    js = Path(root) / "build" / "m8" / NET / "map_stades.json"
    if js.exists():
        d = json.loads(js.read_text())
        out = {}
        for s in d["stages"]:
            eq = [num(x) for x in re.findall(r"\d[\d ]*", str(s.get("eq") or ""))]
            out[s["key"]] = (None if s["map"] is None else round(100 * s["map"], 2),
                             *(eq + [None, None])[:2])
        out["_images"] = d["images"]
        return out
    path = _results("map_stades.md", root)
    _, rows = table_with(path, "Stade", "mAP")
    keys = ("flottant", "entier", "csim", "carte")
    out = {}
    for k, r in zip(keys, rows):
        eq = re.findall(r"(\d[\d ]*\d|\d) / (\d[\d ]*\d|\d)", r[2])
        out[k] = (num(r[1]), *((num(eq[0][0]), num(eq[0][1])) if eq else (None, None)))
    m = re.search(r"\((\d+) images\)", need(path).read_text())
    out["_images"] = int(m.group(1)) if m else None
    return out


def stage(key, root):
    m = read_stages(root)[key][0]
    if m is None:
        raise MissingSource(f"stage {key} not yet measured")
    return m


def t_stages(root):
    st = read_stages(root)
    labels = {"flottant": ("Float (fused BN, NumPy float32)", "measured"),
              "entier": ("Integer, Python (`IntNetwork`, bit-exact)", "measured"),
              "csim": ("FPGA, C-sim (ARM driver, sim backend, PC)", "csim"),
              "carte": ("FPGA, KV260 (uio backend)", "measured")}
    rows = []
    for k, (label, status) in labels.items():
        m, eq, n = st.get(k, (None, None, None))
        if k == "flottant":
            same = "—"
        elif k == "entier":
            same = "reference"
        elif eq is not None:
            same = f"{fmt(eq)} / {fmt(n)} identical images"
        else:
            same = "to be measured"
        rows.append([label, m, same, status if m is not None else "—"])
    return {"header": ["Stage", "mAP", "Equality with the integer model", "Status"],
            "rows": rows, "nd": [None, 2, None, None]}


def identical(root):
    _, eq, _ = read_stages(root)["csim"]
    if eq is None:
        raise MissingSource("C-sim stage without equality count")
    return eq


def images(root):
    n = read_stages(root)["_images"]
    if n is None:
        raise MissingSource("image count absent")
    return n


EN = [  # phrases of the French result tables → English (M19 glossary)
    ("redimensionnement direct", "direct resize"), ("Pillow bilinéaire", "Pillow bilinear"),
    ("référence T3.4, ", ""), ("celui de l'entraînement", "the training one"),
    ("canaux valides seulement", "valid channels only"),
    ("requantification", "requantization"), ("pliage de L00", "L00 folding"),
    ("tuiles 14 pour les convs poolées (noyau M10)", "14×14 tiles for pooled convs (M10 kernel)"),
    ("borne calcul du noyau M10", "compute bound of the M10 kernel"),
    ("borne calcul M6 (chargements gratuits)", "compute bound of M6 (free loads)"),
    ("point roofline KV260", "KV260 roofline point"), ("idéal", "ideal"),
    ("efficacité", "efficiency"), ("élagué à", "pruned at"), ("élagué", "pruned"),
    ("complet", "full"), ("lot ", "batch "), ("puissances de 2", "powers of 2"),
    ("hétérogène", "heterogeneous"), ("ce travail (projection C-sim)", "this work (C-sim projection)"),
    ("M6 : ports", "M6: ports"), ("par canal", "per channel"),
]


def en(s):
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s).replace("**", "")
    for fr, e in EN:
        s = s.replace(fr, e)
    return s.strip()


def t_float(root):
    _, rows = table_with(_results("map_float.md", root), "Prétraitement", "Interpolation")
    return {"header": ["Preprocessing", "Interpolation", "mAP"],
            "rows": [[en(r[0]), en(r[1]), num(r[2])] for r in rows], "nd": [None, None, 2]}


def darknet_published(root):
    m = re.search(r"publiée par Darknet[^*]*\*\*(\d+,\d+)\*\*",
                  need(_results("map_float.md", root)).read_text())
    if not m:
        raise MissingSource("published Darknet mAP not found in map_float.md")
    return num(m.group(1))


def t_int8(root):
    _, rows = table_with(_results("map_int8.md", root), "Prétraitement", "Écart")
    return {"header": ["Preprocessing", "Float (fused BN)", "Integer INT8", "Gap"],
            "rows": [[en(r[0]), num(r[1]), num(r[2]), round(num(r[2]) - num(r[1]), 2)]
                     for r in rows],
            "nd": [None, 2, 2, 2], "sign": [False, False, False, True]}


def sensitivity(root):
    """(layer, gap) of the most sensitive layer of map_int8.md."""
    _, rows = table_with(_results("map_int8.md", root), "Couche quantifiée")
    rows = [r for r in rows if re.match(r"\**L\d", r[0])]
    r = min(rows, key=lambda r: num(r[2]))
    return r[0].replace("*", ""), num(r[2])


FORMATS = [  # (label, method, file, row words, column of mAP)
    ("INT8 per channel (reference)", "PTQ", "map_int8.md", None),
    ("uniform6 (±31)", "PTQ", "req_yolo.md", ("uniform6",)),
    ("mixed6 (powers of 2)", "PTQ, direct projection", "req_yolo.md", ("mixed6", "PTQ")),
    ("mixed6", "ADMM, 600 iterations, then projection", "req_yolo.md", ("mixed6", "ADMM")),
    ("w4a4", "PTQ (power-of-2 steps)", "quant_4bits.md", ("w4a4 PTQ",)),
    ("w4a4", "QAT, 600 iterations", "quant_4bits.md", ("w4a4 QAT",)),
    ("INT8 + hardware post-processing", "unsorted NMS, Q4 boxes", "postproc_hw.md",
     ("matériel : tout entier",)),
]


def formats(root):
    ref = stage("entier", root)
    out = []
    for label, method, name, words in FORMATS:
        if words is None:
            m = ref
        else:
            header, rows = table_with(_results(name, root), "mAP")
            col = header.index("mAP")
            r = next((r for r in rows if all(w in " ".join(r[:2]) for w in words)), None)
            if r is None:
                raise MissingSource(f"{name}: no row {words}")
            m = num(r[col])
        out.append((label, method, m, round(m - ref, 2)))
    return out


def t_formats(root):
    return {"header": ["Weights / activations", "Method", "mAP", "Gap to INT8"],
            "rows": [[a, b, m, None if i == 0 else g]
                     for i, (a, b, m, g) in enumerate(formats(root))],
            "nd": [None, None, 2, 2], "sign": [False, False, False, True]}


def fmt_map(label, method):
    return lambda root: next(m for a, b, m, _ in formats(root) if a == label and b == method)


def perf(root):
    """{net: {gmac, mcycles, ms, fps, gops, eff}} of the projection table of mesures.md."""
    _, rows = table_with(_results("mesures.md", root), "Réseau", "GOPS")
    keys = ("gmac", "mcycles", "ms", "fps", "gops", "eff")
    return {r[0]: dict(zip(keys, (num(c) for c in r[1:]))) for r in rows}


def t_perf(root):
    rows = [[f"`{n}`", p["gmac"], p["mcycles"], p["ms"], p["fps"], p["gops"], p["eff"],
             "projection"] for n, p in perf(root).items()]
    return {"header": ["Network", "GMAC", "Mcycles", "ms", "img/s", "GOPS",
                       "MAC efficiency (%)", "Status"],
            "rows": rows, "nd": [None, 3, 2, 1, 1, 1, 1, None]}


def cascade(root):
    _, rows = table_with(_results("rapport.md", root), "Scénario", "Mcycles")
    return [[en(r[0]).replace("… ", "... "), num(r[1]), num(r[2]), num(r[3]), num(r[4]),
             num(r[5])] for r in rows]


def t_cascade(root):
    return {"header": ["Kernel configuration (Tiny-YOLOv2)", "Mcycles", "ms", "img/s", "GOPS",
                       "MAC efficiency (%)"],
            "rows": cascade(root), "nd": [None, 2, 1, 1, 1, 1], "status": "projection"}


def cascade_row(words, col):
    def read(root):
        for r in cascade(root):
            if words in r[0]:
                return r[col]
        raise MissingSource(f"rapport.md: no scenario '{words}'")
    return read


def streaming(root):
    text = need(_results("streaming.md", root)).read_text()
    plans = re.findall(r"\*\*([\d.]+) img/s\*\*.*?latence ≈ ([\d.]+) ms\s*\n- DSP (\d+) /",
                       text)
    if len(plans) < 2:
        raise MissingSource("streaming.md: 8-bit and 4-bit plans not found")
    return [{"fps": float(f), "latency": float(lat), "dsp": int(d)} for f, lat, d in plans[:2]]


def t_bench(root):
    rows = []
    with need(_results("benchmarks.csv", root)).open() as f:
        for r in csv.DictReader(f):
            ours = r["travail"].startswith("ce travail")

            fps, ms, gops, w = (fmt(num(r[k]), len(r[k].partition(".")[2])) if r[k] else "—"
                                for k in ("fps", "latence_ms", "gops", "puissance_w"))
            rows.append([en(r["travail"]), en(r["modele"]), r["fpga"], en(r["format"]),
                         fps, ms, gops, w,
                         "projection" if ours else "published"])
    return {"header": ["Work", "Model", "FPGA", "Format", "img/s", "ms", "GOPS", "W", "Status"],
            "rows": rows}


def macs(net, what):
    def read(root):
        sys.path.insert(0, str(Path(root) / "python"))
        sys.path.insert(0, str(Path(root) / "tools"))
        from count_macs import table
        from yolo.models.specs import NETWORKS

        rows = table(NETWORKS[net])
        if what == "macs":
            return round(sum(r[6] for r in rows) / 1e9, 2)
        return round(sum(r[5] for r in rows) / 1e6, 2)
    return read


DATASETS = ("VOC", "VisDrone", "KITTI", "FLIR", "ExDark", "CrowdHuman")
SWEEPS = Path("docs") / "tasks" / "resultats-balayages.md"


def sweeps(root):
    """{dataset: (train images, best run, batch, subset, mAP)} of the first sweep table of
    each dataset section of resultats-balayages.md (rows sorted best first)."""
    out, section, n_train = {}, None, None
    text = need(Path(root) / SWEEPS).read_text().splitlines()
    for i, line in enumerate(text):
        if line.startswith("## "):
            section = line[3:].strip() if line[3:].strip() in DATASETS else None
        elif section and line.startswith("### Balayage"):
            m = re.search(r"(\d[\d ]*\d) images", line)
            n_train = num(m.group(1)) if m else None
        elif (section and section not in out and line.startswith("| run | lot")
              and i + 2 < len(text)):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            col = next(j for j, h in enumerate(header) if h.startswith("mAP"))
            r = [c.strip() for c in text[i + 2].strip().strip("|").split("|")]
            out[section] = (n_train, r[0].replace("*", ""), num(r[1]), r[2], num(r[col]))
    if not out:
        raise MissingSource(f"{SWEEPS}: no sweep table")
    return out


def t_sweeps(root):
    rows = []
    for ds, (n, run, batch, subset, m) in sweeps(root).items():
        metric = "AP@[.5:.95] (COCO)" if ds == "FLIR" else "mAP, 11 points"
        rows.append([ds, n, f"`{run}`", batch, "all" if subset == "tout" else subset, metric,
                     m, "tier-R"])
    return {"header": ["Dataset", "Training images", "Best run", "Batch", "Training subset",
                       "Metric", "Score (50 images)", "Status"],
            "rows": rows, "nd": [None, 0, None, 0, None, None, 2, None]}


def sweep_best(ds):
    return lambda root: sweeps(root)[ds][4]


def makefile_dumps(root):
    text = need(Path(root) / "Makefile").read_text()
    imgs = re.search(r"^DUMP_IMAGES = (.*)$", text, re.MULTILINE).group(1).split()
    nets = re.search(r"^QNETS = (.*)$", text, re.MULTILINE).group(1).split()
    return len(imgs) * len(nets)


def hwpp_cases(root):
    m = re.search(r"\(`tb_post`[^:]*: ([\d ]+) cas",
                  need(_results("postproc_hw.md", root)).read_text())
    if not m:
        raise MissingSource("postproc_hw.md: number of tb_post cases not found")
    return num(m.group(1))


# ------------------------------------------------------------------------- registry

@dataclass
class Key:
    read: Callable
    source: str
    command: str
    status: str
    nd: int | None = None
    sign: bool = False


R = "results/"
CMD_STAGES = "make m8-int bench-sim map-stades"
CLES: dict[str, Key] = {
    # §3 networks
    "model.v2.gmacs": Key(macs("tiny-yolov2-voc", "macs"), "python/yolo/models/specs.py",
                          "make count-macs", "measured", 2),
    "model.v2.mparams": Key(macs("tiny-yolov2-voc", "params"), "python/yolo/models/specs.py",
                            "make count-macs", "measured", 2),
    "model.v3.gmacs": Key(macs("tiny-yolov3-voc", "macs"), "python/yolo/models/specs.py",
                          "make count-macs", "measured", 2),
    "model.v3.mparams": Key(macs("tiny-yolov3-voc", "params"), "python/yolo/models/specs.py",
                            "make count-macs", "measured", 2),
    # §4-§5-§8 mAP on VOC2007 test
    "map.voc.images": Key(images, R + "map_stades.md", CMD_STAGES, "measured", 0),
    "map.voc.float": Key(lambda r: stage("flottant", r), R + "map_stades.md", CMD_STAGES,
                         "measured", 2),
    "map.voc.int8": Key(lambda r: stage("entier", r), R + "map_stades.md", CMD_STAGES,
                        "measured", 2),
    "map.voc.csim": Key(lambda r: stage("csim", r), R + "map_stades.md", CMD_STAGES, "csim", 2),
    "map.voc.drop": Key(lambda r: round(stage("entier", r) - stage("flottant", r), 2),
                        R + "map_stades.md", CMD_STAGES, "measured", 2, True),
    "map.voc.identical": Key(identical, R + "map_stades.md", "make bench-sim map-stades",
                             "csim", 0),
    "map.voc.darknet": Key(darknet_published, R + "map_float.md", "make eval-float",
                           "published", 1),
    "map.voc.float.table": Key(t_float, R + "map_float.md", "python tools/eval_voc.py "
                               "--resize stretch|letterbox [--interp darknet]", "measured"),
    "map.voc.int8.table": Key(t_int8, R + "map_int8.md", "make eval-int", "measured"),
    "quant.sensitive.layer": Key(lambda r: sensitivity(r)[0], R + "map_int8.md",
                                 "python tools/eval_quant.py (fake-quant sweep)", "tier-R"),
    "quant.sensitive.gap": Key(lambda r: sensitivity(r)[1], R + "map_int8.md",
                               "python tools/eval_quant.py (fake-quant sweep)", "tier-R", 2,
                               True),
    "map.stades": Key(t_stages, R + "map_stades.md", CMD_STAGES, "measured"),
    # §6 verification
    "verif.dumps": Key(makefile_dumps, "Makefile (DUMP_IMAGES × QNETS)", "make golden-check",
                       "measured", 0),
    "verif.hwpp.cases": Key(hwpp_cases, R + "postproc_hw.md", "make csim-gcc", "csim", 0),
    # §7-§8 performance (projection until the board, T17.19)
    "perf.kv260": Key(t_perf, R + "mesures.md", "make bench-report", "projection"),
    "perf.kv260.v2.ms": Key(lambda r: perf(r)["tiny-yolov2-voc"]["ms"], R + "mesures.md",
                            "make perf-model bench-report", "projection", 1),
    "perf.kv260.v2.fps": Key(lambda r: perf(r)["tiny-yolov2-voc"]["fps"], R + "mesures.md",
                             "make perf-model bench-report", "projection", 1),
    "perf.kv260.v2.gops": Key(lambda r: perf(r)["tiny-yolov2-voc"]["gops"], R + "mesures.md",
                              "make perf-model bench-report", "projection", 1),
    "perf.kv260.v2.eff": Key(lambda r: perf(r)["tiny-yolov2-voc"]["eff"], R + "mesures.md",
                             "make perf-model bench-report", "projection", 1),
    "perf.kv260.v3.ms": Key(lambda r: perf(r)["tiny-yolov3-coco"]["ms"], R + "mesures.md",
                            "make perf-model bench-report", "projection", 1),
    "perf.cascade": Key(t_cascade, R + "rapport.md", "make perf-model", "projection"),
    "perf.kv260.m6.ms": Key(cascade_row("M6: ports", 2), R + "rapport.md", "make perf-model",
                            "projection", 1),
    "perf.kv260.bound.ms": Key(cascade_row("compute bound of the M10", 2), R + "rapport.md",
                               "make perf-model", "projection", 1),
    "perf.kv260.roofline.ms": Key(cascade_row("roofline", 2), R + "rapport.md",
                                  "make roofline", "projection", 1),
    # §9 extensions
    "map.formats": Key(t_formats, R + "req_yolo.md, quant_4bits.md, postproc_hw.md",
                       "python tools/eval_quant.py --variants …", "measured"),
    "map.formats.uniform6": Key(fmt_map("uniform6 (±31)", "PTQ"), R + "req_yolo.md",
                                "python tools/eval_quant.py", "measured", 2),
    "map.formats.mixed6": Key(fmt_map("mixed6 (powers of 2)", "PTQ, direct projection"),
                              R + "req_yolo.md", "python tools/eval_quant.py", "measured", 2),
    "map.formats.admm": Key(fmt_map("mixed6", "ADMM, 600 iterations, then projection"),
                            R + "req_yolo.md", "python tools/train.py --admm", "measured", 2),
    "map.formats.w4a4.ptq": Key(fmt_map("w4a4", "PTQ (power-of-2 steps)"),
                                R + "quant_4bits.md", "python tools/quant_lowbit.py",
                                "measured", 2),
    "map.formats.w4a4.qat": Key(fmt_map("w4a4", "QAT, 600 iterations"), R + "quant_4bits.md",
                                "python tools/train.py --qat w4a4", "measured", 2),
    "map.formats.hwpp": Key(fmt_map("INT8 + hardware post-processing", "unsorted NMS, Q4 boxes"),
                            R + "postproc_hw.md",
                            "python tools/eval_quant.py --variants int,int-hwpp", "measured", 2),
    "stream.w8.fps": Key(lambda r: streaming(r)[0]["fps"], R + "streaming.md",
                         "python tools/stream_model.py", "estimate", 1),
    "stream.w8.latency": Key(lambda r: streaming(r)[0]["latency"], R + "streaming.md",
                             "python tools/stream_model.py", "estimate", 1),
    "stream.w8.dsp": Key(lambda r: streaming(r)[0]["dsp"], R + "streaming.md",
                         "python tools/stream_model.py", "estimate", 0),
    "stream.w4.fps": Key(lambda r: streaming(r)[1]["fps"], R + "streaming.md",
                         "python tools/stream_model.py", "estimate", 1),
    # §2 related work
    "benchmarks": Key(t_bench, R + "benchmarks.csv", "make bench-report", "published"),
    # §10 beyond VOC
    "balayages.meilleurs": Key(t_sweeps, str(SWEEPS), "make harvest", "tier-R"),
    **{f"balayage.{ds.lower()}.meilleur.map": Key(sweep_best(ds), str(SWEEPS), "make harvest",
                                                  "tier-R", 2) for ds in DATASETS},
}


def collect(data, root=ROOT, keys=None, rev=None, date=None):
    """Refresh `data` from the sources; return (changed keys, {key: why not refreshed})."""
    keys = keys or CLES
    rev, date = rev or git_rev(root), date or today()
    changed, stale = [], {}
    for k, spec in keys.items():
        try:
            v = spec.read(root)
        except (MissingSource, OSError) as e:
            stale[k] = str(e)
            continue
        if isinstance(v, float):
            v = round(v, 6)
        entry = {"value": v, "status": spec.status, "source": spec.source,
                 "command": spec.command}
        if spec.nd is not None:
            entry["nd"] = spec.nd
        if spec.sign:
            entry["sign"] = True
        old = data.get(k, {})
        if {x: old.get(x) for x in entry} != entry:
            entry.update(rev=rev, date=date)
            data[k] = entry
            changed.append(k)
    return changed, stale


def render_file(article=ARTICLE, data_path=DATA, root=ROOT, figures=None, suivi_path=SUIVI,
                force_rev=False):
    """Rewrite the blocks of `article`; the stored revision changes only with the content."""
    data = load(data_path)
    text = Path(article).read_text()
    new = render(text, data, figures, article, root, suivi_path)
    if new != text or force_rev or "_rev" not in data:
        data["_rev"] = {"rev": git_rev(root), "date": today()}
        save(data, data_path)
        new = render(text, data, figures, article, root, suivi_path)
    if new != text:
        Path(article).write_text(new)
        return True
    return False


PDF_CSS = """
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 10.5pt; line-height: 1.4;
       max-width: none; margin: 0; padding: 0; }
h1 { font-size: 17pt; } h2 { font-size: 13pt; margin-top: 1.4em; }
img { max-width: 100%; max-height: 22cm; display: block; margin: 0.6em auto; }
table { border-collapse: collapse; font-size: 8.5pt; margin: 0.6em 0; }
th, td { border: 1px solid #999; padding: 2px 5px; }
code { font-size: 9pt; } blockquote { color: #444; border-left: 3px solid #ccc;
       margin-left: 0; padding-left: 0.8em; }
@page { size: A4; margin: 18mm 16mm; }
"""
VERSIONED_PDF = ARTICLE.with_suffix(".pdf")


def _chromium():
    """Chromium on PATH, or the one of Playwright (PLAYWRIGHT_BROWSERS_PATH)."""
    found = next((shutil.which(b) for b in ("chromium", "chromium-browser", "google-chrome")
                  if shutil.which(b)), None)
    pw = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if found is None and pw:
        found = next((str(c) for c in sorted(Path(pw).glob("chromium-*/chrome-linux*/chrome"))),
                     None)
    return found


def export(article=ARTICLE, out=BUILD, versioned=VERSIONED_PDF):
    """pandoc → out/article.html, then out/article.pdf with a LaTeX engine, or else by
    printing the HTML with headless Chromium; the PDF is copied to `versioned`
    (docs/article/article.pdf, the export kept in git; build/ is not versioned)."""
    if not shutil.which("pandoc"):
        print("pandoc not installed: no export (apt install pandoc)")
        return 0
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Block markers removed: at the start of a line pandoc reads them as HTML blocks and
    # cuts the paragraph around an inline number.
    src = out / "article.md"
    src.write_text(re.sub(r"<!-- (article:[a-z]+(?::\S+?)?|/article|python -m tools\.figures \S+) -->",
                          "", Path(article).read_text()))
    adir = str(Path(article).resolve().parent)
    base = ["pandoc", str(src), "--from", "gfm", "--resource-path", adir]
    css = out / "article.css"
    css.write_text(PDF_CSS)
    html = out / "article.html"
    title = Path(article).read_text().splitlines()[0].lstrip("# ").strip()
    subprocess.run(base + ["--standalone", "--metadata", f"pagetitle={title}", "--css",
                           str(css), "-o", str(html)], cwd=adir, check=True)
    print(f"→ {_rel(html)}")
    pdf = out / "article.pdf"
    engine = next((e for e in ("xelatex", "lualatex", "pdflatex") if shutil.which(e)), None)
    if engine:
        subprocess.run(base + ["--pdf-engine", engine, "-o", str(pdf)], cwd=adir, check=True)
    elif _chromium():
        subprocess.run([_chromium(), "--headless", "--no-sandbox", "--disable-gpu",
                        "--no-pdf-header-footer", "--allow-file-access-from-files",
                        f"--print-to-pdf={pdf}", html.as_uri()],
                       check=True, capture_output=True)
    else:
        print("no LaTeX engine (xelatex, lualatex, pdflatex) nor Chromium: PDF skipped")
        return 0
    print(f"→ {_rel(pdf)}")
    if versioned:
        shutil.copyfile(pdf, versioned)
        print(f"→ {_rel(versioned)}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tools.article", description=__doc__.split(
        "\n\n")[0])
    ap.add_argument("action", nargs="?", choices=("collect", "render", "all", "pdf"))
    ap.add_argument("--check", action="store_true",
                    help="fail if a block differs from render, a key or a figure is missing, "
                    "or a result section quotes a number outside a block")
    ap.add_argument("--article", type=Path, default=ARTICLE)
    ap.add_argument("--data", type=Path, default=DATA)
    args = ap.parse_args(argv)
    if args.check:
        problems = check(args.article.read_text(), load(args.data), article=args.article)
        for p in problems:
            print(f"{_rel(args.article)}: {p}")
        if problems:
            print("regenerate: make article")
            return 1
        print(f"{_rel(args.article)}: {len(parse(args.article.read_text()))} blocks up to date")
        return 0
    if args.action is None:
        ap.error("collect, render, all, pdf or --check")
    if args.action in ("collect", "all", "pdf"):
        data = load(args.data)
        changed, stale = collect(data)
        save(data, args.data)
        print(f"{len(changed)} keys changed" + (f": {', '.join(changed)}" if changed else ""))
        for k, why in stale.items():
            print(f"  not refreshed  {k} ({why}; value of {_rel(args.data)} kept)")
    if args.action in ("render", "all", "pdf"):
        try:
            done = render_file(args.article, args.data)
        except ArticleError as e:
            print(f"{_rel(args.article)}: {e}")
            return 1
        print(f"→ {_rel(args.article)}" + ("" if done else " (unchanged)"))
    if args.action == "pdf":
        return export(args.article)
    return 0


if __name__ == "__main__":
    sys.exit(main())
