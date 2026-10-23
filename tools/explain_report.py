"""M36: tables of `results/explain.md` from `build/explain/*.json` (`tools/explain.py report`).

Each section is written between `<!-- explain:<name>:begin -->` and `<!-- explain:<name>:end
-->`; the text around them is written by hand. A missing JSON leaves its section as it is.
The miss examples (one Grad-CAM map per cause, 5 examples, Tiny-YOLOv2) go to
`results/explain/misses_<cause>.png`.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "explain"
DOC = ROOT / "results" / "explain.md"
PNG = ROOT / "results" / "explain"
SECTIONS = ("sanity", "stages", "intattr", "evaluate", "misses", "concepts", "record", "cka")


def _load(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _m(v, f=".2f"):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return format(float(np.mean(v)), f) if v else "–"


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _short(net):
    return net.replace("tiny-yolo", "").replace("-coco", "").replace("-voc", "")


# ---------------------------------------------------------------------- sections
def sanity(d):
    rows = []
    for net, res in d.items():
        for m in res[0]["methods"] if res else []:
            r = [x["methods"][m] for x in res]
            comp = [x.get("completeness") for x in r]
            rand = {k: [x["rand"][k] for x in r] for k in r[0]["rand"]}
            ok = np.mean(rand["all"]) < 0.3
            rows.append([_short(net), m, len(r),
                         _m(comp, ".3f") if comp[0] is not None else "–",
                         *(_m(v) for v in rand.values()),
                         "pass" if ok else "**fail**", _m([x["passes"] for x in r], ".0f"),
                         _m([x["seconds"] for x in r], ".1f")])
    return _table(["net", "method", "images", "IG completeness error", "ρ top conv",
                   "ρ top 3", "ρ all convs", "sanity (ρ < 0.3)", "passes", "s / map"], rows)


def stages(d):
    rows = []
    for net, res in d.items():
        for m in ("drise", "occlusion"):
            rows.append([_short(net), m, len(res),
                         _m([x[m]["spearman_float_int"] for x in res]),
                         _m([x[m]["passes"] for x in res], ".0f"),
                         _m([x[m]["seconds"]["float"] for x in res], ".0f"),
                         _m([x[m]["seconds"]["int"] for x in res], ".0f")])
    g = []
    for net, res in d.items():
        gg = [x["golden"] for x in res if "golden" in x]
        if gg:
            g.append([_short(net), len(gg), gg[0]["masks"],
                      f"{sum(x['weights_equal'] for x in gg)} / {len(gg)}",
                      f"{sum(x['map_equal'] for x in gg)} / {len(gg)}",
                      _m([x["seconds"] for x in gg], ".0f")])
    return (_table(["net", "method", "images", "Spearman float / int", "passes / map",
                    "s / map float", "s / map int"], rows)
            + "\n\nInteger against golden, the same masks:\n\n"
            + _table(["net", "images", "masks", "weights equal", "maps equal (bit for bit)",
                      "s / map golden (1 process)"], g))


def intattr(d):
    rows, lay = [], []
    for net, res in d.items():
        rows.append([_short(net), len(res),
                     f"{sum(x['exact_bad'] for x in res)} / {sum(x['exact_checked'] for x in res)}",
                     _m([x["total_eps"] for x in res]), _m([x["total_zplus"] for x in res]),
                     _m([x["spearman_int_float_lrp"] for x in res]),
                     _m([x["spearman_int_lrp_gradcam"] for x in res]),
                     _m([x["spearman_zplus_gradcam"] for x in res]),
                     _m([x["seconds_int"] for x in res], ".1f")])
        for rule in ("eps", "zplus"):
            per = defaultdict(list)
            for x in res:
                for k, v in x[f"conservation_{rule}"].items():
                    per[int(k)].append(v)
            lay.append([_short(net), rule, " ".join(f"L{k}:{np.mean(v):.2f}"
                                                    for k, v in sorted(per.items()))])
    return (_table(["net", "images", "Σ contributions ≠ acc − qb", "relevance at input (ε)",
                    "(z⁺)", "ρ int LRP / float LRP", "ρ int LRP / Grad-CAM",
                    "ρ z⁺ / Grad-CAM", "s / map"], rows)
            + "\n\nRelevance kept per conv (Σ R_in / Σ R_out, mean over the images):\n\n"
            + _table(["net", "rule", "per layer"], lay))


def evaluate(d):
    rows, bins, rank = [], [], defaultdict(list)
    for net, res in d.items():
        if not res:
            continue
        for key in res[0]["rows"]:
            r = [x["rows"][key] for x in res]
            ins, dele = np.mean([x["insertion"] for x in r]), np.mean([x["deletion"] for x in r])
            rank[key].append(ins - dele)
            rows.append([_short(net), key, len(r), _m([x["pointing"] for x in r]),
                         _m([x["energy"] for x in r]), f"{dele:.3f}", f"{ins:.3f}",
                         _m([x["passes"] for x in r], ".0f"), _m([x["seconds"] for x in r],
                                                                  ".1f")])
            by = defaultdict(list)
            for x in res:
                by[x["bin"]].append(x["rows"][key]["pointing"])
            bins.append([_short(net), key] + [f"{np.mean(by[b]):.2f} ({len(by[b])})" if by[b]
                                              else "–" for b in ("small", "medium", "large")])
    order = sorted(rank, key=lambda k: -np.mean(rank[k]))
    return (_table(["net", "method @ stage", "images", "pointing", "energy in box",
                    "deletion AUC ↓", "insertion AUC ↑", "passes", "s / map"], rows)
            + "\n\nPointing rate per box size (images):\n\n"
            + _table(["net", "method @ stage", "small", "medium", "large"], bins)
            + "\n\nRanking by insertion − deletion, mean over the networks: "
            + ", ".join(f"{k} ({np.mean(rank[k]):.3f})" for k in order) + ".")


def misses(d):
    from yolo.explain.misses import CAUSES

    rows = []
    for net, res in d.items():
        for st in ("float", "int"):
            c = [x[st] for x in res]
            n = sum(v is not None for v in c)
            rows.append([_short(net), st, len(c), f"{n} ({n / max(len(c), 1):.0%})"]
                        + [f"{sum(v == k for v in c) / n:.0%}" if n else "–" for k in CAUSES])
    return _table(["net", "stage", "persons", "missed"] + list(CAUSES), rows)


def concepts(d):
    from yolo.data.voc import VOC_CLASSES

    top = [[VOC_CLASSES[int(k)], *(f"{ch} ({v:.3f})" for ch, v in d["top"][k])]
           for k in sorted(d["top"], key=int)]
    tc = [[k, v["images"], f"{v['score']:.2f}", f"{v['random_mean']:.2f}", f"{v['t']:.2f}",
           f"{v['p']:.3g}", "yes" if v["p"] < 0.05 else "no"] for k, v in d["tcav"].items()]
    return (f"{d['net']}, L{d['layer']}, {d['images']} VOC test images. Share of the "
            f"channels matching no class (best IoU < 0.04): **{d['unmatched_share']:.0%}**.\n\n"
            + _table(["class", "channel 1 (IoU)", "channel 2", "channel 3"], top)
            + "\n\nTCAV for the class person (20 random sets of 30 images):\n\n"
            + _table(["concept", "images", "TCAV score", "random-random mean", "t", "p",
                      "significant (p < 0.05)"], tc))


def record(d):
    rows = []
    for run, r in d.items():
        rows.append([run, r["images"], r["epochs"], f"{r['overhead']:.1%}", r["never_learned"],
                     r["forgotten"], r.get("corrupted", "–"),
                     f"{r['auroc_never_learned']:.2f}" if "auroc_never_learned" in r else "–",
                     f"{r['auroc_mean_loss']:.2f}" if "auroc_mean_loss" in r else "–"])
    return _table(["run", "images", "epochs", "recorder overhead", "never learned", "forgotten",
                   "corrupted", "AUROC never learned", "AUROC mean loss"], rows)


def cka(d):
    rows = []
    for net, per in d["float_int"].items():
        low = min(per.items(), key=lambda kv: kv[1])
        below = [k for k, v in per.items() if v < 0.99]
        rows.append([_short(net), len(per), f"{low[1]:.4f} (L{low[0]})",
                     ", ".join(f"L{k}" for k in below) or "none"])
    out = (f"Probe: {d['probe']} VOC test images, maps pooled over 2 × 2 cells.\n\n"
           + _table(["net", "convs", "min CKA float / int (layer)", "layers < 0.99"], rows))
    if "checkpoints" in d:
        c = d["checkpoints"]
        m = np.array(c["cka_to_last"])
        out += ("\n\nCKA of each conv to the last checkpoint, run `" + c["run"] + "`:\n\n"
                + _table(["iteration"] + [f"conv {k}" for k in range(m.shape[1])],
                         [[it, *(f"{v:.2f}" for v in row)] for it, row in zip(c["its"], m)]))
    return out


# --------------------------------------------------------------------- examples
def miss_examples(d, per_cause=5, net="tiny-yolov2-voc"):
    import explain as X

    from yolo.data import datasets
    from yolo.explain import gradient as G
    from yolo.explain.targets import miss_target

    if not d or net not in d:
        return []
    by_id = {s["id"]: s for s in datasets.load("voc", split="2007:test")}
    groups = defaultdict(list)
    for r in d[net]:
        if r["float"] is not None:
            groups[r["float"]].append(r)
    paths = []
    model = X.models(net)[0]
    for cause, rows in groups.items():
        maps = {}
        for r in rows[:per_cause]:
            x, _ = X.load_image(by_id[r["id"]]["image"])
            t = miss_target(model.net, r["box"], X.person_class(net))
            maps[f"{r['id']}"] = (x, G.gradcam(model, x, t)["map"], r["box"])
        paths.append(_strip(maps, PNG / f"misses_{cause}.png", f"{net}: {cause} (Grad-CAM)"))
    return paths


def _strip(items, path, title):
    from tools.figures.style import plt

    from yolo.explain.maps import normalize

    p = plt()
    fig, axes = p.subplots(1, len(items), figsize=(3 * len(items), 3.2), squeeze=False)
    for ax, (name, (x, m, box)) in zip(axes[0], items.items()):
        img = x[0].transpose(1, 2, 0)
        ax.imshow(img)
        ax.imshow(normalize(m), cmap="jet", alpha=0.5)
        cx, cy, bw, bh = box
        h, w = img.shape[:2]
        ax.add_patch(p.Rectangle(((cx - bw / 2) * w, (cy - bh / 2) * h), bw * w, bh * h,
                                 fill=False, color="w", lw=1.5))
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    p.close(fig)
    return path


# ------------------------------------------------------------------------- write
def splice(doc, name, body):
    begin, end = f"<!-- explain:{name}:begin -->", f"<!-- explain:{name}:end -->"
    block = f"{begin}\n{body}\n{end}"
    if begin in doc:
        a, b = doc.index(begin), doc.index(end) + len(end)
        return doc[:a] + block + doc[b:]
    return doc.rstrip() + "\n\n" + block + "\n"


def report():
    doc = DOC.read_text() if DOC.exists() else "# Decision explanation (M36)\n"
    funcs = {"sanity": sanity, "stages": stages, "intattr": intattr, "evaluate": evaluate,
             "misses": misses, "concepts": concepts, "record": record, "cka": cka}
    for name in SECTIONS:
        d = _load(name)
        if d is None:
            print(f"{name}: no build/explain/{name}.json, section kept")
            continue
        doc = splice(doc, name, funcs[name](d))
        print(f"{name}: written")
    DOC.write_text(doc)
    for p in miss_examples(_load("misses")):
        print(p.relative_to(ROOT))
