"""Cross-dataset analysis of the dataset statistics (M16): reads the executed notebooks
`notebooks/<ds>/<ds>_stats.ipynb` and compares the datasets with each other.

    python -m tools.notebooks.jeux                         # summary table
    python -m tools.notebooks.jeux --json build/notebooks/jeux/jeux.json
    python -m tools.notebooks.jeux --figures build/notebooks/jeux

Source: the versioned outputs of the `_stats` notebooks (M14 rule), i.e. the Markdown
tables written by the `md_*` functions of `tools/data_stats.py`. The `stats.json` files
stay on the runtime that produced them (Colab), so nothing is recomputed and nothing is
typed by hand. Stdlib + NumPy (ADR 0001).

Lost targets are split in two causes: objects whose best anchor belongs to no head of the
cfg (orphans: Tiny-YOLOv3 COCO masks leave anchor 0 unused) and objects that fall on the
same cell and anchor as another one (collisions). Both are dropped by `build_targets`.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from tools.notebooks import NB_DIR
from tools.notebooks.balayages import _cells, _executed, _rel

GROUND = ("voc", "kitti", "crowdhuman", "exdark", "flir", "coco", "asl", "trashnet", "pets",
          "prewid")
AERIAL = ("visdrone", "auair", "dronevehicle", "hituav", "uavdt")
ORDER = GROUND + AERIAL  # order of tables and figures: ground views, then drone views
LABELS = {"voc": "VOC", "coco": "COCO", "visdrone": "VisDrone", "kitti": "KITTI",
          "flir": "FLIR", "exdark": "ExDark", "crowdhuman": "CrowdHuman", "auair": "AU-AIR",
          "dronevehicle": "DroneVehicle", "hituav": "HIT-UAV", "uavdt": "UAVDT",
          "asl": "ASL", "trashnet": "TrashNet", "pets": "Pet",
          "prewid": "PREWID"}
VOC = "voc"  # reference dataset: INT8 calibration images (M4) and published weights
SLOTS = 256  # yolo_post slots (M9.1)
VOC_LUMINANCE = 112.0  # VOC2007 mean luminance (INT8 calibration, M4) when VOC is not loaded
SHIFT = ("classes", "area_in", "w_in", "h_in", "aspect", "per_image")

# Table kind ← start of its header row (first cells, as written by tools/data_stats.py).
KINDS = [("synthese", "| partie | images | objets | difficult"),
         ("classes", "| classe | "),
         ("geometrie", "| repère | petits"),
         ("densite", "| partie | objets/image moy."),
         ("images", "| partie | résolutions"),
         ("ancres", "| ancres | w,h"),
         ("qualite", "| partie | retirées"),
         ("ecart", "| grandeur (train ↔ test)"),
         ("couverture", "| split | classes du modèle"),
         ("eclairage", "| éclairage |")]


# ------------------------------------------------------------------------- parsing

def num(text):
    """French-formatted number → float: `40 570`, `70,9 %`, `0,6029`; `—` → None."""
    t = re.sub(r"[\s \xa0%]", "", str(text)).replace(",", ".")
    if t in ("", "—", "-"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _ints(text):
    """All integers of a cell, French thousands separators included (`2 146 / 56 612`)."""
    return [int(re.sub(r"[\s \xa0]", "", m))
            for m in re.findall(r"\d{1,3}(?:[  \xa0]\d{3})+|\d+", text)]


def _heads(text):
    """`13x13 57,7 % · 26x26 25,5 %` → {'13x13': 57.7, '26x26': 25.5}."""
    return {m.group(1): num(m.group(2))
            for m in re.finditer(r"(\d+x\d+)\s+([\d\s \xa0,]+)\s*%", text)}


def _rows(block):
    """Markdown table → (header cells, [row cells])."""
    lines = [ln for ln in block.splitlines() if ln.startswith("|")]
    split = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    return split[0], [r for r in split[2:]]


def _blocks(text):
    """[(table lines, first non-blank line after it)]: a table starts at a row followed by
    `|---`, so two tables printed back to back (two outputs of a cell) stay apart."""
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
            j = i + 2
            while j < len(lines) and lines[j].startswith("|") and not (
                    j + 1 < len(lines) and lines[j + 1].startswith("|---")):
                j += 1
            k = j
            while k < len(lines) and not lines[k].strip():
                k += 1
            after = lines[k] if k < len(lines) and not lines[k].startswith("|") else ""
            out.append(("\n".join(lines[i:j]), after))
            i = j
        else:
            i += 1
    return out


def _tables(cells):
    """[(kind, header, rows, text after the table)] of the Markdown outputs, in order."""
    out = []
    for _, text, _ in cells:
        for b, after in _blocks(text):
            kind = next((k for k, h in KINDS if b.startswith(h)), None)
            if kind is None:
                continue
            if kind == "classes" and "objets (images)" not in b.splitlines()[0]:
                continue  # class mapping (VOC / COCO) or per-class AP table
            head, rows = _rows(b)
            out.append((kind, head, rows, after))
    return out


def _whole(rows):
    """Row of the whole dataset (`ensemble`), else the last row."""
    return next((r for r in rows if r[0] == "ensemble"), rows[-1])


def _synthese(rows):
    r = _whole(rows)
    med, mx = _ints(r[5])[:2]
    over = _ints(r[9])
    heads = _heads(r[7])
    return {"images": num(r[1]), "objects": num(r[2]), "difficult": num(r[3]),
            "empty": num(r[4]), "median": med, "max": mx, "small": num(r[6]),
            "cell": heads, "lost": num(r[8]), "over256": over[0] if over else None,
            "splits": [x[0] for x in rows if x[0] not in ("ensemble", "groupe train")]}


def _classes(head, rows, after):
    splits = [h.split(" : ")[0] for h in head[1:-2]]
    names, counts, per_split, side, small = [], [], [], [], []
    for r in rows:
        c = [_ints(x)[0] if _ints(x) else 0 for x in r[1:-2]]
        names.append(r[0])
        per_split.append(c)
        counts.append(sum(c))
        side.append(num(r[-2]))
        small.append(num(r[-1]))
    m = re.search(r"fréquente : ([\d\s \xa0]+)", after)
    return {"names": names, "counts": counts, "splits": splits, "per_split": per_split,
            "side": side, "small": small, "ratio": num(m.group(1)) if m else None}


def _geometrie(rows, after):
    out = {}
    for r in rows:
        key = ("orig" if r[0].startswith("image") else r[0].split()[0])
        out[key] = {"small": num(r[1]), "medium": num(r[2]), "large": num(r[3]),
                    "side": num(r[4]), "cell": _heads(r[5]), "h8": num(r[6])}
        if key != "orig":
            out["size"] = r[0].split()[1]
    return out


def _densite(rows):
    r = _whole(rows)
    lost, objects = _ints(r[5])[:2]
    heads = {}
    for m in re.finditer(r"(\d+x\d+) : ([\d\s \xa0]+) / ([\d\s \xa0]+?)(?: ·|$)", r[6]):
        heads[m.group(1)] = {"lost": _ints(m.group(2))[0], "objects": _ints(m.group(3))[0]}
    assigned = sum(h["objects"] for h in heads.values())
    collide = sum(h["lost"] for h in heads.values())
    return {"mean": num(r[1]), "median": num(r[2]), "max": num(r[3]),
            "over256": _ints(r[4])[0], "lost": lost, "objects": objects, "heads": heads,
            "orphan": objects - assigned, "collide": collide, "images_hit": num(r[7])}


def _images(rows):
    r = _whole(rows)
    res = [(int(w), int(h), _ints(n)[0]) for w, h, n in
           re.findall(r"(\d+)×(\d+) \(([\d\s \xa0]+)\)", r[1])]
    return {"resolutions": res, "distinct": num(r[2]), "channels": int(num(r[4])),
            "mean": [num(x) for x in r[5].split("/")], "std": [num(x) for x in r[6].split("/")],
            "luminance": num(r[7])}


def _anchors(rows):
    iou, wh = {}, {}
    for r in rows:
        name = r[0]
        if name.startswith("VOC (tiny-yolov2"):
            key = "voc2"
        elif name.startswith("VOC"):
            key = "voc3"
        elif name.startswith("COCO"):
            key = "coco"
        elif name.startswith("cfg"):
            key = "cfg"
        else:
            m = re.search(r"k = (\d+)", name)
            key = f"k{m.group(1)}" if m else name
        iou[key] = num(r[2])
        wh[key] = [tuple(int(v) for v in p.split(",")) for p in r[1].strip("`").split()]
    return {"iou": iou, "wh": wh}


def _qualite(rows):
    r = _whole(rows)
    tiny = _ints(r[3])
    regions = {m.group(1): _ints(m.group(2))[0]
               for m in re.finditer(r"([A-Za-z_][\w-]*) ([\d\s \xa0]+)", r[5])}
    return {"removed": num(r[1]), "clipped": num(r[2]),
            "tiny_orig": tiny[0] if tiny else None, "tiny_in": tiny[1] if len(tiny) > 1 else None,
            "duplicates": num(r[4]), "regions": regions}


def _couverture(rows):
    """Objects-weighted share of objects with a VOC / COCO class, over the listed splits."""
    tot = {}
    for r in rows:
        covered, part = num(r[2]), num(r[3])
        if covered is None or not part:
            continue
        c, n = tot.get(r[1], (0.0, 0.0))
        tot[r[1]] = (c + covered, n + covered * 100 / part)
    return {k: 100 * c / n for k, (c, n) in tot.items() if n}


def parse_stats(path):
    """Everything the comparison needs, read from an executed `_stats` notebook."""
    path = Path(path)
    cells = _cells(path)
    ds = path.parent.name
    text = "\n".join(t for _, t, _ in cells)
    m = re.search(r"révision (\w+)", text) or re.search(r"@ (\w+) ;", text)
    cfg = re.search(r"; cfg (\S+)\s*$", text, re.M)
    d = {"dataset": ds, "label": LABELS.get(ds, ds), "notebook": _rel(path),
         "family": "aerial" if ds in AERIAL else "ground",
         "rev": m.group(1) if m else None, "net": cfg.group(1) if cfg else None,
         "fallback": bool(re.search(r"cfg \S+ absente", text))}
    for kind, head, rows, after in _tables(cells):
        if not rows:
            continue
        if kind == "geometrie" and "geometrie" in d:
            d["geometrie_native"] = _geometrie(rows, after)  # KITTI: its own input size
            continue
        if kind in d:
            continue  # the summary table is printed again at the end of the notebook
        if kind == "synthese":
            d[kind] = _synthese(rows)
        elif kind == "classes":
            d[kind] = _classes(head, rows, after)
        elif kind == "geometrie":
            d[kind] = _geometrie(rows, after)
        elif kind == "densite":
            d[kind] = _densite(rows)
        elif kind == "images":
            d[kind] = _images(rows)
        elif kind == "ancres":
            d[kind] = _anchors(rows)
        elif kind == "qualite":
            d[kind] = _qualite(rows)
        elif kind == "ecart":
            d[kind] = {r[0]: num(r[1]) for r in rows}
        elif kind == "couverture":
            d[kind] = _couverture(rows)
        elif kind == "eclairage":
            d[kind] = [(r[0], num(r[1]), num(r[2])) for r in rows]
    return _derive(d)


def _derive(d):
    """Indicators computed from the parsed tables (shares in %, entropy, anchor gain)."""
    s, c = d.get("synthese", {}), d.get("classes")
    n = s.get("objects") or 0
    if c and c["counts"]:
        p = np.array(c["counts"], dtype=np.float64)
        p = p[p > 0] / p.sum()
        c["entropy"] = float(-(p * np.log(p)).sum() / np.log(len(c["counts"]))) \
            if len(c["counts"]) > 1 else 1.0
    dn = d.get("densite")
    if dn and dn["objects"]:
        dn["orphan_pct"] = 100 * dn["orphan"] / dn["objects"]
        dn["collide_pct"] = 100 * dn["collide"] / dn["objects"]
    q = d.get("qualite")
    if q and n:
        q["clipped_pct"] = 100 * (q["clipped"] or 0) / n
        q["removed_pct"] = None if q["removed"] is None else 100 * q["removed"] / n
        q["tiny_in_pct"] = 100 * (q["tiny_in"] or 0) / n
        q["regions_n"] = sum(q["regions"].values())
    a = d.get("ancres")
    if a:
        best = max((v for k, v in a["iou"].items() if k.startswith("k")), default=None)
        a["best_k"] = best
        a["gain"] = best - a["iou"]["cfg"] if best is not None and "cfg" in a["iou"] else None
    im = d.get("images")
    if im and im["resolutions"]:
        w, h, _ = max(im["resolutions"], key=lambda r: r[2])
        im["main"] = (w, h)
        im["scale"] = 416 / max(w, h)  # letterbox factor of the most common resolution
    return d


# -------------------------------------------------------------------------- loading

def find_stats(nb_dir=NB_DIR):
    """{dataset: path} of the `*_stats.ipynb`, executed or not, in ORDER."""
    found = {p.parent.name: p for p in Path(nb_dir).glob("*/*_stats.ipynb")}
    return {k: found[k] for k in sorted(found, key=lambda k: (ORDER + (k,)).index(k))}


def _error(path):
    """The error a notebook stopped on, or None."""
    nb = json.loads(Path(path).read_text())
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                return f"{o.get('ename', 'error')}: {o.get('evalue', '')[:120]}"
    return None


def load_all(nb_dir=NB_DIR, datasets=None):
    """({dataset: parsed stats}, {dataset: reason it is left out}). A run that stopped
    early without error is kept (`partial`) if its tables are there: they are written by
    the computation cells, before the question and summary cells."""
    data, pending = {}, {}
    for ds, p in find_stats(nb_dir).items():
        if datasets and ds not in datasets:
            continue
        err = _error(p)
        if err:
            pending[ds] = f"{_rel(p)} ({err})"
            continue
        d = parse_stats(p)
        if not all(k in d for k in ("synthese", "geometrie", "densite", "images", "ancres")):
            pending[ds] = f"{_rel(p)} (not executed)"
            continue
        d["partial"] = not _executed(p)
        data[ds] = d
    return data, pending


# ------------------------------------------------------------------------- tables

def fmt(v, d=1):
    """English number: decimal point, comma thousands separator; None → —."""
    if v is None:
        return "—"
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and d == 0):
        return f"{int(round(v)):,}"
    return f"{v:,.{d}f}"


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def summary_table(data):
    """One row per dataset: size, density, object size, lost targets, pixels, anchors."""
    rows = []
    for d in data.values():
        s, g, dn = d["synthese"], d.get("geometrie", {}), d.get("densite", {})
        im, a = d.get("images", {}), d.get("ancres", {})
        lb = g.get("letterbox", {})
        rows.append([d["label"], d["family"], len(d.get("classes", {}).get("names", [])),
                     fmt(s["images"], 0), fmt(s["objects"], 0), f"{s['median']} / {s['max']}",
                     fmt(lb.get("side"), 0), fmt(s["small"]),
                     fmt(s["cell"].get("26x26")), fmt(s["lost"]),
                     fmt(dn.get("collide_pct"), 2), im.get("channels", "—"),
                     fmt(im.get("luminance"), 0), fmt(a.get("iou", {}).get("cfg"), 3),
                     fmt(a.get("best_k"), 3)])
    return _table(["dataset", "view", "classes", "images", "objects", "objects/img (med / max)",
                   "median side (px)", "small < 32² (%)", "< fine cell (%)",
                   "lost targets (%)", "of which collisions (%)", "channels", "luminance",
                   "IoU cfg anchors", "IoU k-means"], rows)


def provenance_table(data):
    """Where each row comes from: notebook, revision, anchors used for the lost targets."""
    rows = [[d["label"], f"[{Path(d['notebook']).name}]({Path(d['notebook']).relative_to('notebooks').as_posix()})", d["rev"] or "—",
             "partial" if d.get("partial") else "complete", d["net"] or "—",
             "yes" if d["fallback"] else "no", ", ".join(d["synthese"]["splits"])]
            for d in data.values()]
    return _table(["dataset", "notebook", "revision", "run", "anchors (cfg)",
                   "fallback to COCO cfg", "splits"], rows)


# ------------------------------------------------------------------- observations

def _top(data, key, n=3, reverse=True, positive=False):
    vals = [(d["label"], v) for d in data.values()
            if (v := key(d)) is not None and (v > 0 or not positive)]
    return sorted(vals, key=lambda t: t[1], reverse=reverse)[:n]


def _list(pairs, unit="", d=1):
    def one(v):
        return f"< {10 ** -d:g}" if 0 < v < 10 ** -d / 2 else fmt(v, d)
    return ", ".join(f"{lab} ({one(v)}{unit})" for lab, v in pairs)


def _get(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def observations(data):
    """{section: one data-driven sentence} — recomputed from the loaded datasets."""
    obs = {}
    sz = _top(data, lambda d: d["synthese"]["objects"])
    sm = _top(data, lambda d: d["synthese"]["objects"], reverse=False)
    obs["taille"] = (f"Most objects: {_list(sz, d=0)}; fewest: {_list(sm, d=0)}. "
                     f"Densest images (median objects per image): "
                     f"{_list(_top(data, lambda d: d['synthese']['median']))}.")
    small = _top(data, lambda d: d["synthese"]["small"])
    big = _top(data, lambda d: d["synthese"]["small"], reverse=False)
    obs["objets"] = (f"After letterbox to 416, the share of small objects (< 32²) is highest for "
                     f"{_list(small, ' %')} and lowest for {_list(big, ' %')}. Median side "
                     f"seen by the network: from "
                     f"{_list(_top(data, lambda d: _get(d, 'geometrie', 'letterbox', 'side'), 1, False), ' px', 0)} "
                     f"to {_list(_top(data, lambda d: _get(d, 'geometrie', 'letterbox', 'side'), 1), ' px', 0)}.")
    fine = _top(data, lambda d: d["synthese"]["cell"].get("26x26"))
    h8 = _top(data, lambda d: _get(d, "geometrie", "letterbox", "h8"))
    gain = _top(data, lambda d: (_get(d, "geometrie", "letterbox", "cell", "26x26") or 0)
                - (_get(d, "geometrie", "stretch", "cell", "26x26") or 0), 2)
    obs["cellules"] = (f"Objects smaller than a 16 px cell of the fine 26×26 head: "
                       f"{_list(fine, ' %')}. Objects under 8 px high: {_list(h8, ' %')}. "
                       f"Stretch instead of letterbox helps most where images are far from "
                       f"square: {_list(gain, ' points')}.")
    orph = _top(data, lambda d: _get(d, "densite", "orphan_pct"))
    col = _top(data, lambda d: _get(d, "densite", "collide_pct"))
    over = [d["label"] for d in data.values() if (d["synthese"]["over256"] or 0) > 0]
    obs["densite"] = (f"Most lost targets come from anchors, not crowding: orphans (best anchor "
                      f"in no head, e.g. anchor 0 = 10×14 px with the COCO masks 3,4,5 / 1,2,3) reach {_list(orph, ' %')}, while real cell + anchor "
                      f"collisions stay at {_list(col, ' %', 2)}. "
                      + (f"Images above the {SLOTS} `yolo_post` slots: "
                         f"{', '.join(over)}." if over else
                         f"No image exceeds the {SLOTS} `yolo_post` slots."))
    ratio = _top(data, lambda d: _get(d, "classes", "ratio"))
    ent = _top(data, lambda d: _get(d, "classes", "entropy"), reverse=False)
    obs["classes"] = (f"Largest most / least frequent class ratios: {_list(ratio, '×', 0)}. "
                      f"Least balanced (normalised entropy, 1 = uniform): {_list(ent, '', 2)}.")
    ref = _get(data.get(VOC, {}), "images", "luminance")
    lum = _top(data, lambda d: _get(d, "images", "luminance"), reverse=False)
    gray = [d["label"] for d in data.values() if _get(d, "images", "channels") == 1]
    obs["images"] = (f"Darkest datasets: {_list(lum, '', 0)}"
                     + (f", against {fmt(ref, 0)} for VOC (INT8 calibration images)" if ref else "")
                     + "." + (f" Single-channel (thermal) images: {', '.join(gray)}; "
                              "they are replicated to 3 channels at the input." if gray else ""))
    g = _top(data, lambda d: _get(d, "ancres", "gain"))
    low = _top(data, lambda d: _get(d, "ancres", "iou", "cfg"), reverse=False)
    obs["ancres"] = (f"Lowest mean IoU with the anchors in use: {_list(low, '', 3)}. Re-clustering "
                     f"(k-means) gains most for {_list(g, '', 3)} IoU.")
    sh = []
    for d in data.values():
        e = d.get("ecart") or {}
        if e:
            k = max(e, key=lambda q: e[q] or 0)
            sh.append((d["label"], k, e[k]))
    sh.sort(key=lambda t: t[2] or 0, reverse=True)
    obs["ecart"] = ("Largest train ↔ test gaps (total variation): "
                    + ", ".join(f"{lab} on {k} ({fmt(v, 3)})" for lab, k, v in sh[:3])
                    + "; values under 0.05 mean the test split looks like the training split.")
    voc = _top(data, lambda d: _get(d, "couverture", "VOC"), reverse=False)
    coco = _top(data, lambda d: _get(d, "couverture", "COCO"), reverse=False)
    obs["couverture"] = (f"Lowest share of objects the published weights can name: VOC classes "
                         f"{_list(voc, ' %')}; COCO classes {_list(coco, ' %')}.")
    cl = _top(data, lambda d: _get(d, "qualite", "clipped_pct"), positive=True)
    rm = _top(data, lambda d: _get(d, "qualite", "removed_pct"), positive=True)
    tiny = _top(data, lambda d: _get(d, "qualite", "tiny_in_pct"), positive=True)
    obs["qualite"] = (f"Clipped boxes: {_list(cl, ' %', 2) or 'none'}; removed by the loader: "
                      f"{_list(rm, ' %', 2) or 'none'}; boxes under 2 px at the input: "
                      f"{_list(tiny, ' %', 2) or 'none'}. The other datasets are cleaner.")
    sc = difficulty(data)
    order = sorted(sc["score"], key=lambda k: sc["score"][k], reverse=True)
    obs["difficulte"] = ("Ranking by mean normalised difficulty (hardest first): "
                         + ", ".join(f"{data[k]['label']} ({sc['score'][k]:.2f})" for k in order)
                         + ".")
    return obs


# Difficulty indicators: (key, label, getter); higher = harder for Tiny-YOLO on the target.
INDICATORS = [
    ("small", "small < 32² (%)", lambda d: d["synthese"]["small"]),
    ("fine", "< fine cell (%)", lambda d: d["synthese"]["cell"].get("26x26")),
    ("collide", "cell collisions (%)", lambda d: _get(d, "densite", "collide_pct")),
    ("dense", "median objects/img", lambda d: d["synthese"]["median"]),
    ("imbal", "class imbalance (log₁₀ ratio)",
     lambda d: float(np.log10(r)) if (r := _get(d, "classes", "ratio")) else None),
    ("anchors", "1 − IoU k-means", lambda d: 1 - v if (v := _get(d, "ancres", "best_k")) else None),
    ("light", "luminance gap to VOC",
     lambda d: abs(v - VOC_LUMINANCE) if (v := _get(d, "images", "luminance")) is not None
     else None),
]


def difficulty(data):
    """Raw indicator values, min–max normalised columns and their mean per dataset."""
    global VOC_LUMINANCE
    VOC_LUMINANCE = _get(data.get(VOC), "images", "luminance") or VOC_LUMINANCE
    raw = {k: {ds: fn(d) for ds, d in data.items()} for k, _, fn in INDICATORS}
    norm = {}
    for k, col in raw.items():
        v = [x for x in col.values() if x is not None]
        lo, hi = (min(v), max(v)) if v else (0, 1)
        norm[k] = {ds: None if x is None else (x - lo) / (hi - lo) if hi > lo else 0.0
                   for ds, x in col.items()}
    score = {ds: float(np.mean([norm[k][ds] for k in norm if norm[k][ds] is not None]))
             for ds in data}
    return {"raw": raw, "norm": norm, "score": score}


def conclusions(data, pending):
    """Short list of cross-dataset conclusions for the embedded Tiny-YOLO target."""
    sc = difficulty(data)["score"]
    hard = sorted(sc, key=sc.get, reverse=True)
    lines = []
    aerial = [d for d in data.values() if d["family"] == "aerial"]
    ground = [d for d in data.values() if d["family"] == "ground"]
    if aerial and ground:
        ma = np.mean([d["synthese"]["small"] for d in aerial])
        mg = np.mean([d["synthese"]["small"] for d in ground])
        lines.append(f"- **Drone views are a different regime**: on average {ma:.0f} % small "
                     f"objects after letterbox 416, against {mg:.0f} % for ground views (VOC "
                     f"{_get(data.get(VOC), 'synthese', 'small') or 0:.0f} %). Few of their "
                     "objects reach the 13×13 head; the 26×26 head (or a finer one, or a "
                     "larger input) carries the detection.")
    orph = [d for d in data.values() if (_get(d, "densite", "orphan_pct") or 0) > 5]
    if orph:
        lines.append("- **Anchors before capacity**: with the COCO anchors and masks, "
                     + ", ".join(f"{d['label']} {d['densite']['orphan_pct']:.0f} %" for d in orph)
                     + " of the objects have their best anchor in no head and are never trained. "
                     "Dataset-specific anchors (k-means, M11 cfg) fix this before any change "
                     "of architecture; real collisions stay small.")
    dark = [d for d in data.values() if (_get(d, "images", "luminance") or 999) < 60]
    if dark:
        lines.append("- **INT8 calibration**: " + ", ".join(
            f"{d['label']} (luminance {d['images']['luminance']:.0f})" for d in dark)
            + " is far from the VOC calibration images; calibrate the quantisation scales on "
            "the dataset itself.")
    over = [d["label"] for d in data.values() if (d["synthese"]["over256"] or 0) > 0]
    lines.append("- **`yolo_post` slots**: " + (
        f"{', '.join(over)} have images with more than {SLOTS} objects: the hardware "
        "post-processing would drop detections there." if over else
        f"no dataset exceeds {SLOTS} objects per image."))
    lines.append("- **Hardest to easiest** (mean normalised difficulty): "
                 + " > ".join(data[k]["label"] for k in hard) + ".")
    fb = [d["label"] for d in data.values() if d["fallback"]]
    if fb:
        lines.append(f"- **Caveat**: {', '.join(fb)} were run without their M11 cfg; lost "
                     "targets and \"cfg\" anchors use Tiny-YOLOv3 COCO. Re-run after "
                     "`tools/m11.sh <ds>-prep` to see the fine-tuning cfg.")
    if pending:
        lines.append("- **Not analysed**: " + "; ".join(f"{k}: {v}" for k, v in pending.items()))
    return "\n".join(lines)


# ---------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tools.notebooks.jeux",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", default=None, help="comma-separated datasets")
    ap.add_argument("--json", type=Path, default=None, help="write all parsed data")
    ap.add_argument("--figures", type=Path, default=None, help="write the PNG figures there")
    args = ap.parse_args(argv)
    data, pending = load_all(datasets=args.datasets.split(",") if args.datasets else None)
    if not data:
        raise SystemExit("no executed notebook in notebooks/*/*_stats.ipynb")
    print(summary_table(data))
    for ds, why in pending.items():
        print(f"pending: {ds} — {why}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"data": data, "pending": pending}, indent=1,
                                        ensure_ascii=False, default=float) + "\n")
        print(f"→ {args.json}")
    if args.figures:
        from tools.figures import jeux as FJ

        for fn in FJ.PLOTS.values():
            for p in fn(data, args.figures):
                print(f"→ {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
