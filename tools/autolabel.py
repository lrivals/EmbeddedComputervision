"""Photo auto-labelling with manual correction (M22): open-vocabulary boxes → YOLO txt.

    python tools/autolabel.py                       # local web app, http://127.0.0.1:7860
    python tools/autolabel.py --model owlv2-base --device cpu
    python tools/autolabel.py --list                # labellers, device, cached weights
    python tools/autolabel.py --folder ~/Pictures/garden --classes "cat, dog" \\
        --out build/autolabel/garden [--conf 0.3] [--resume] [--recursive]   # no display

Classes: comma-separated, their order gives the ids (`classes.txt`); `name=prompt` sends
another text to the model (`tvmonitor=tv monitor`). Presets: `voc`, `coco`,
`dataset:<name>`. A result folder is `labels/<rel>.txt`, `classes.txt`, `autolabel.json`
(tools/autolabel_io.py); docs/autolabel.md is the user guide.

The server listens on 127.0.0.1 only and has no share link: the photos stay on the PC.
gradio, torch and transformers are imported here and in tools/autolabel_models.py only,
never by `yolo/`.
"""

import argparse
import importlib.util
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

import autolabel_io as lio  # noqa: E402
import autolabel_models as lm  # noqa: E402

OUT_DIR = ROOT / "build" / "autolabel"
EXTRAS = ("torch", "transformers", "gradio", "gradio_image_annotation")
MERGE_IOU = 0.5     # Re-run: a new prediction overlapping a manual box is dropped
SAVE_EVERY = 20     # images between two writes of autolabel.json in a folder run


# ---------------------------------------------------------------- classes

def presets():
    from yolo.data.datasets import COCO_CLASSES, DATASETS
    from yolo.data.voc import VOC_CLASSES

    out = {"voc": VOC_CLASSES, "coco": COCO_CLASSES}
    out.update({f"dataset:{k}": d.classes for k, d in sorted(DATASETS.items())})
    return out


def preset_text(name):
    try:
        return ", ".join(presets()[name])
    except KeyError:
        raise ValueError(f"unknown preset {name!r}: {', '.join(presets())}") from None


def session_classes(classes=None, preset=None):
    """`--classes` / `--preset` → (names, prompts)."""
    text = classes if classes else preset_text(preset) if preset else ""
    names, prompts = lm.parse_classes(text)
    if not names:
        raise ValueError("no class: --classes \"person, dog\" or --preset voc")
    return names, prompts


# ---------------------------------------------------------------- folder batch (T22.3)

def label_folder(labeller, files, root, out, names, prompts, conf=None, resume=False,
                 floor=lm.FLOOR):
    """Label `files` (under `root`) into `out`: `labels/<rel>.txt`, `classes.txt`,
    `autolabel.json`. A generator: yields `(done, total, rel path)` after each image, so
    that the CLI and the "Folder" tab (T22.6) show progress; returns the provenance.

    `resume`: images whose txt exists are skipped. The raw predictions of an image already
    in `autolabel.json` with the same SHA-1 and model are reused without running the model.
    """
    out, root = Path(out), Path(root)
    conf = lm.default_conf(labeller.name) if conf is None else conf
    (out / "labels").mkdir(parents=True, exist_ok=True)
    prov = lio.read_provenance(out)
    if prov is not None and (prov["classes"], prov["model"]) != (list(names), labeller.name):
        if resume:
            raise ValueError(f"{out}: labelled by {prov['model']} with classes "
                             f"{prov['classes']}; --resume needs the same, or a new --out")
        prov = None  # stable ids: another class list or model starts a new session
    if prov is None:
        prov = lio.new_provenance(labeller.name, labeller.version, names, prompts, conf)
    prov["conf"] = conf
    cache = {r["sha1"]: r["raw"] for r in prov["images"].values()}
    lio.write_classes(out / lio.CLASSES, names)
    files = list(files)
    for i, path in enumerate(files, 1):
        rel = Path(path).relative_to(root)
        txt = out / "labels" / rel.with_suffix(".txt")
        if not (resume and txt.exists()):
            try:
                img = lio.load_image(path)
                digest = lio.sha1(path)
            except Exception as exc:  # noqa: BLE001 — unreadable file: listed and skipped
                prov["unreadable"] = [u for u in prov["unreadable"]
                                      if u["path"] != rel.as_posix()]
                prov["unreadable"].append({"path": rel.as_posix(), "error": str(exc)[:200]})
                yield i, len(files), rel
                continue
            rec = prov["images"].get(rel.as_posix())
            if rec is None or rec["sha1"] != digest:
                raw = cache.get(digest)
                if raw is None:
                    raw = lio.raw_record(*labeller.predict(img, (names, prompts), floor))
                    cache[digest] = raw
                rec = lio.image_record(digest, img.size, raw)
                prov["images"][rel.as_posix()] = rec
            lio.export_labels(rec, conf, txt)
        if i % SAVE_EVERY == 0:
            lio.write_provenance(out, prov)
        yield i, len(files), rel
    lio.write_provenance(out, prov)
    return prov


def run_folder(labeller, files, root, out, names, prompts, conf=None, resume=False):
    """`label_folder` to the end; the provenance."""
    gen = label_folder(labeller, files, root, out, names, prompts, conf, resume)
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value


def relabel(out, conf):
    """Labels of a result folder rewritten at `conf` (raw predictions + corrections)."""
    out = Path(out)
    prov = lio.read_provenance(out)
    prov["conf"] = conf
    for rel, rec in prov["images"].items():
        lio.export_labels(rec, conf, out / "labels" / Path(rel).with_suffix(".txt"))
    lio.write_provenance(out, prov)
    return prov


# ---------------------------------------------------------------- --list (T22.0)

def missing_extras():
    def found(m):
        try:
            return importlib.util.find_spec(m) is not None
        except (ImportError, ValueError):  # broken install: counted as missing
            return False
    return [m for m in EXTRAS if not found(m)]


def print_list(device):
    missing = missing_extras()
    print(f"device: {lm.pick_device(device)}" + ("" if "torch" not in missing else
                                                    " (torch missing)"))
    print("labellers:")
    for name in [*lm.REGISTRY, *lm.YOLO_MODELS]:
        lic = "project" if name.startswith("yolo:") else lm.REGISTRY[name].licence
        ready = "cached" if lm.weights_ready(name) else (
            "missing: make get-weights" if name.startswith("yolo:") else "download on first run")
        star = "*" if name == lm.DEFAULT else " "
        print(f" {star}{name:22s} {lic:11s} {ready:22s} {lm.NOTES[name]}")
    print(f"weights cache: {lm.os.environ['HF_HOME']}")
    if missing:
        print(f"missing extras: {', '.join(missing)} → {lm.EXTRA_HINT}")


# ---------------------------------------------------------------- web app (T22.4-T22.6)

LABELLERS = {}  # (model, device) → labeller, loaded once and shared by both tabs


def get_labeller(name, device, cfg=None, dataset=None):
    key = (name, device)
    if key not in LABELLERS:
        LABELLERS[key] = lm.get_labeller(name, device, cfg, dataset)
    return LABELLERS[key]


def color_hex(c):
    from detect import COLORS

    return "#%02x%02x%02x" % COLORS[c % len(COLORS)]


def to_pixels(box, w, h):
    """Normalised `(cx, cy, w, h)` → integer `(xmin, ymin, xmax, ymax)` in the image."""
    cx, cy, bw, bh = box
    return (int(np.clip(round((cx - bw / 2) * w), 0, w)), int(np.clip(round((cy - bh / 2) * h), 0, h)),
            int(np.clip(round((cx + bw / 2) * w), 0, w)), int(np.clip(round((cy + bh / 2) * h), 0, h)))


def annotator_boxes(record, names, conf):
    """Labels of `record` at `conf` → (boxes of the editor, raw index of each box)."""
    w, h = record["width"], record["height"]
    boxes, scores, labels, idx = lio.labels_of(record, conf)
    out = []
    for b, s, c in zip(boxes, scores, labels):
        x0, y0, x1, y1 = to_pixels(b, w, h)
        out.append({"xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1, "label": names[c],
                    "color": color_hex(c)})
    return out, [int(i) for i in idx]


def apply_edit(record, names, shown, boxes):
    """Boxes of the editor after a user edit → `record` corrections. A shown raw box still
    present unchanged stays a raw box; a shown raw box gone (deleted, moved, other class) is
    `removed`; every other box is `manual`. Idempotent: the editor value written by
    `annotator_boxes` gives back the same record."""
    w, h = record["width"], record["height"]
    key = {}
    for (b, i) in zip(*shown):
        if i >= 0:
            key.setdefault((b["xmin"], b["ymin"], b["xmax"], b["ymax"], b["label"]), []).append(i)
    kept, manual = set(), []
    for b in boxes or []:
        t = (int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"]), b.get("label"))
        if key.get(t):
            kept.add(key[t].pop())
        elif b.get("label") in names and t[2] > t[0] and t[3] > t[1]:
            manual.append([(t[0] + t[2]) / 2 / w, (t[1] + t[3]) / 2 / h, (t[2] - t[0]) / w,
                           (t[3] - t[1]) / h, names.index(b["label"])])
    gone = {i for b, i in zip(*shown) if i >= 0} - kept
    record["removed"] = sorted(set(record["removed"]) | gone)
    record["manual"] = manual
    record["edited"] = bool(record["removed"] or record["manual"])
    return record


def merge_manual(record, raw, names_old, names_new, iou_thr=MERGE_IOU):
    """Re-run: new raw predictions under the manual corrections. The manual boxes are kept
    (score 1); a new box is dropped when it overlaps a manual box, or a box of the same
    class deleted by hand. `removed` then indexes the new predictions (none)."""
    from yolo.infer.boxes import iou

    old = record["raw"]
    gone = [i for i in record["removed"] if names_old[old["labels"][i]] in names_new]
    ref = np.array([m[:4] for m in record["manual"]]
                   + [old["boxes"][i] for i in gone], dtype=np.float64).reshape(-1, 4)
    ref_cls = np.array([-1] * len(record["manual"])  # manual: any class
                       + [names_new.index(names_old[old["labels"][i]]) for i in gone])
    boxes = np.asarray(raw["boxes"], dtype=np.float64).reshape(-1, 4)
    if len(ref) and len(boxes):
        same = (ref_cls[None, :] < 0) | (ref_cls[None, :] == np.asarray(raw["labels"])[:, None])
        keep = ~((iou(boxes, ref) > iou_thr) & same).any(axis=1)
        raw = {k: [v for v, k_ in zip(raw[k], keep) if k_] for k in raw}
    record.update(raw=raw, removed=[])
    record["edited"] = bool(record["manual"])
    return record


def folder_edit(out, rel, shown, boxes):
    """An edit in the "Folder" tab: corrections of image `rel` of result folder `out`,
    its txt and `autolabel.json` rewritten. Returns the record, or None if nothing changed."""
    prov = lio.read_provenance(out)
    rec = prov["images"][rel]
    before = (list(rec["removed"]), list(rec["manual"]))
    apply_edit(rec, prov["classes"], shown, boxes)
    if (rec["removed"], rec["manual"]) == before:
        return None
    lio.export_labels(rec, prov["conf"], Path(out) / "labels" / Path(rel).with_suffix(".txt"))
    lio.write_provenance(out, prov)
    return rec


def remap_manual(record, old, new):
    """Manual boxes of class list `old` → ids of `new`; boxes of a dropped class go."""
    record["manual"] = [m[:4] + [new.index(old[int(m[4])])] for m in record["manual"]
                        if old[int(m[4])] in new]
    return record


def build_app(args):
    import gradio as gr
    from gradio_image_annotation import image_annotator

    from detect import draw

    models = [*lm.REGISTRY, *lm.YOLO_MODELS]
    if args.model not in models:
        models.append(args.model)
    work = OUT_DIR / "image"

    def editor(record, names, conf):
        boxes, idx = annotator_boxes(record, names, conf)
        value = {"image": record["path"], "boxes": boxes}
        return value, (boxes, idx)

    def annotator_update(value, names):
        return image_annotator(value=value, label_list=list(names),
                               label_colors=[color_hex(c) for c in range(len(names))])

    def predict(path, model, names, prompts):
        lab = get_labeller(model, args.device, args.cfg, args.dataset)
        img = lio.load_image(path)
        raw = lio.raw_record(*lab.predict(img, (names, prompts), lm.FLOOR))
        return img, raw, lab

    # -------- tab "Image"

    def on_upload(file, classes_text, model, conf):
        if file is None:
            return None, None, None, None
        names, prompts = lm.parse_classes(classes_text)
        if not names:
            raise gr.Error("type the classes first, e.g. person, dog")
        img, raw, lab = predict(file, model, names, prompts)
        # The editor shows the transposed image: save it, so pixels match the boxes.
        work.mkdir(parents=True, exist_ok=True)
        shown_path = work / f"{Path(file).stem}.png"
        img.save(shown_path)
        record = lio.image_record(lio.sha1(file), img.size, raw)
        record.update(path=str(shown_path), name=Path(file).name, model=lab.name,
                      version=lab.version, names=names, prompts=prompts)
        value, shown = editor(record, names, conf)
        return annotator_update(value, names), record, shown, info(record, conf)

    def info(record, conf):
        boxes, scores, labels, _ = lio.labels_of(record, conf)
        n_raw = len(record["raw"]["scores"])
        return (f"{record['name']}: {len(labels)} boxes at conf {conf:.2f} "
                f"({n_raw} raw predictions ≥ {lm.FLOOR}), "
                f"{len(record['manual'])} manual, {len(record['removed'])} removed")

    def on_conf(record, conf):
        if record is None:
            return gr.skip(), None, ""
        value, shown = editor(record, record["names"], conf)
        return value, shown, info(record, conf)

    def on_edit(value, record, shown, conf):
        if record is None or value is None or shown is None:
            return record, gr.skip()
        apply_edit(record, record["names"], shown, value.get("boxes"))
        return record, info(record, conf)

    def on_rerun(record, classes_text, model, conf):
        if record is None:
            raise gr.Error("drop a photo first")
        names, prompts = lm.parse_classes(classes_text)
        img, raw, lab = predict(record["path"], model, names, prompts)
        old = record["names"]
        merge_manual(remap_manual(record, old, names), raw, old, names)
        record.update(names=names, prompts=prompts, model=lab.name, version=lab.version)
        value, shown = editor(record, names, conf)
        return annotator_update(value, names), record, shown, info(record, conf)

    def on_download_labels(record, conf):
        if record is None:
            raise gr.Error("drop a photo first")
        stem = Path(record["name"]).stem
        out = work / stem
        out.mkdir(parents=True, exist_ok=True)
        lio.export_labels(record, conf, out / f"{stem}.txt")
        lio.write_classes(out / lio.CLASSES, record["names"])
        prov = lio.new_provenance(record["model"], record["version"], record["names"],
                                  record["prompts"], conf)
        prov["images"][record["name"]] = {k: v for k, v in record.items()
                                          if k not in ("path", "names", "prompts")}
        lio.write_provenance(out, prov)
        return [str(out / f"{stem}.txt"), str(out / lio.CLASSES), str(out / lio.PROVENANCE)]

    def on_download_image(record, conf):
        if record is None:
            raise gr.Error("drop a photo first")
        boxes, scores, labels, idx = lio.labels_of(record, conf)
        from PIL import Image

        with Image.open(record["path"]) as img:
            dest = work / f"{Path(record['name']).stem}_labelled.png"
            draw(img, boxes, [s if i >= 0 else None for s, i in zip(scores, idx)], labels,
                 record["names"]).save(dest)  # manual boxes: name without score
        return str(dest)

    # -------- tab "Folder"

    def on_folder_run(files, path, recursive, classes_text, model, conf,
                      progress=gr.Progress()):
        names, prompts = lm.parse_classes(classes_text)
        if not names:
            raise gr.Error("type the classes first, e.g. person, dog")
        session = OUT_DIR / time.strftime("%Y%m%d-%H%M%S")
        images = session / "images"
        n_in = gather(files, path, recursive, images)
        if not n_in:
            raise gr.Error("no image: drop files or a zip, or type a folder path")
        lab = get_labeller(model, args.device, args.cfg, args.dataset)
        out = session / "result"
        gen = label_folder(lab, lio.list_images(images, True), images, out, names, prompts,
                           conf)
        progress((0, n_in), desc="labelling")
        while True:
            try:
                done, total, rel = next(gen)
                progress((done, total), desc=str(rel))
            except StopIteration:
                break
        return str(session), *gallery(session, None)

    def gallery(session, selected):
        prov = lio.read_provenance(Path(session) / "result")
        order = lio.review_order(prov["images"])
        items = [(str(Path(session) / "images" / rel),
                  f"{rel} · {prov['images'][rel]['n_boxes']} boxes · "
                  f"best {prov['images'][rel]['best_score']:.2f}"
                  + (" · edited" if prov["images"][rel]["edited"] else ""))
                 for rel in order]
        bad = prov["unreadable"]
        text = (f"{len(order)} images labelled, sorted by needs review "
                f"(no box first, then lowest best score)"
                + (f"; skipped (unreadable): {', '.join(u['path'] for u in bad)}" if bad else ""))
        return items, order, text

    def on_select(session, order, conf, evt: gr.SelectData):
        if not session:
            return gr.skip(), None, None
        rel = order[evt.index]
        prov = lio.read_provenance(Path(session) / "result")
        if prov["conf"] != conf:  # the editor shows and saves at the slider's threshold
            prov = relabel(Path(session) / "result", conf)
        rec = dict(prov["images"][rel], path=str(Path(session) / "images" / rel), rel=rel)
        value, shown = editor(rec, prov["classes"], conf)
        return annotator_update(value, prov["classes"]), rel, shown

    def on_folder_edit(value, session, rel, shown, conf):
        if not (session and rel and value is not None and shown is not None):
            return gr.skip(), gr.skip()
        out = Path(session) / "result"
        rec = folder_edit(out, rel, shown, value.get("boxes"))
        if rec is None:
            return gr.skip(), gr.skip()
        # New shown boxes: the edit is now part of the record.
        prov = lio.read_provenance(out)
        boxes, idx = annotator_boxes(rec, prov["classes"], prov["conf"])
        return (boxes, idx), f"{rel}: saved ({rec['n_boxes']} boxes, edited)"

    def on_download_all(session, conf):
        if not session:
            raise gr.Error("run a folder first")
        out = Path(session) / "result"
        relabel(out, conf)  # the slider applies to the boxes not checked by hand too
        return str(lio.zip_result(out, Path(session) / "labels.zip"))

    # -------- layout

    with gr.Blocks(title="Auto-label") as app:
        gr.Markdown("## Auto-label — open-vocabulary boxes, manual correction, YOLO txt")
        with gr.Row():
            classes = gr.Textbox(label="Classes (comma-separated; order = class id; "
                                       "name=prompt to send another text)",
                                 value=args.classes or (preset_text(args.preset)
                                                        if args.preset else "person, dog"),
                                 scale=4)
            preset = gr.Dropdown(["", *presets()], value="", label="Preset", scale=1)
            model = gr.Dropdown(models, value=args.model, label="Labeller", scale=1)
            conf = gr.Slider(0.05, 0.95, value=args.conf, step=0.01, label="Confidence",
                             scale=1)
        preset.change(lambda p: preset_text(p) if p else gr.skip(), preset, classes)
        model.change(lm.default_conf, model, conf)  # each labeller has its score scale

        with gr.Tab("Image"):
            record, shown = gr.State(None), gr.State(None)
            with gr.Row():
                with gr.Column(scale=1):
                    photo = gr.File(label="Photo (JPEG, PNG, WebP)", file_types=["image"])
                    rerun = gr.Button("Re-run (new classes or labeller; manual boxes kept)")
                    dl_labels = gr.Button("Download labels", variant="primary")
                    dl_image = gr.Button("Download image")
                    files_out = gr.File(label="Files", file_count="multiple")
                with gr.Column(scale=3):
                    ann = image_annotator(label="Boxes: drag to draw, click to select "
                                                "(Del removes), double-click to change class",
                                          sources=[], use_default_label=True,
                                          image_type="filepath")
                    status = gr.Markdown()
            photo.change(on_upload, [photo, classes, model, conf], [ann, record, shown, status])
            conf.change(on_conf, [record, conf], [ann, shown, status])
            ann.change(on_edit, [ann, record, shown, conf], [record, status])
            rerun.click(on_rerun, [record, classes, model, conf], [ann, record, shown, status])
            dl_labels.click(on_download_labels, [record, conf], files_out)
            dl_image.click(on_download_image, [record, conf], files_out)

        with gr.Tab("Folder"):
            session, order, rel, fshown = (gr.State(None), gr.State(None), gr.State(None),
                                           gr.State(None))
            with gr.Row():
                with gr.Column(scale=1):
                    upload = gr.File(label="Images or a zip", file_count="multiple")
                    path = gr.Textbox(label="…or a local folder (copied, never written to)",
                                      placeholder="~/Pictures/garden")
                    recursive = gr.Checkbox(label="Recursive", value=False)
                    run = gr.Button("Label all", variant="primary")
                    dl_all = gr.Button("Download all (zip)")
                    zip_out = gr.File(label="Zip")
                with gr.Column(scale=3):
                    fstatus = gr.Markdown()
                    gal = gr.Gallery(label="Needs review first — click to edit", columns=6,
                                     height=320, allow_preview=False)
                    fann = image_annotator(label="Edits are saved at once", sources=[],
                                           use_default_label=True, image_type="filepath")
            run.click(on_folder_run, [upload, path, recursive, classes, model, conf],
                      [session, gal, order, fstatus])
            gal.select(on_select, [session, order, conf], [fann, rel, fshown])
            fann.change(on_folder_edit, [fann, session, rel, fshown, conf], [fshown, fstatus])
            dl_all.click(on_download_all, [session, conf], zip_out)
    return app


def gather(files, path, recursive, dest):
    """Inputs of the "Folder" tab → copies under `dest`: dropped images, the images of a
    dropped zip, those of a local folder. Returns the image count."""
    dest.mkdir(parents=True, exist_ok=True)
    for f in files or []:
        f = Path(f)
        if f.suffix.lower() == ".zip":
            with zipfile.ZipFile(f) as z:
                for m in z.infolist():
                    p = Path(m.filename)
                    if (not m.is_dir() and p.suffix.lower() in lio.IMAGE_EXT
                            and ".." not in p.parts and not p.is_absolute()):
                        (dest / f.stem / p).parent.mkdir(parents=True, exist_ok=True)
                        (dest / f.stem / p).write_bytes(z.read(m))
        elif f.suffix.lower() in lio.IMAGE_EXT:
            shutil.copy2(f, dest / f.name)
    if path and path.strip():
        src = Path(path.strip()).expanduser()
        if not src.is_dir():
            raise ValueError(f"{src} is not a folder")
        for p in lio.list_images(src, recursive):
            q = dest / src.name / p.relative_to(src)
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)
    return len(lio.list_images(dest, True))


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=lm.DEFAULT,
                    help=f"{', '.join([*lm.REGISTRY, *lm.YOLO_MODELS])} or yolo:ckpt:<npz>")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--list", action="store_true", help="labellers, device, weights; quit")
    ap.add_argument("--classes", help='"person, dog, tvmonitor=tv monitor"')
    ap.add_argument("--preset", help="voc, coco or dataset:<name>")
    ap.add_argument("--conf", type=float, default=None,
                    help="export threshold; default by labeller (T22.7): " + ", ".join(
                        f"{k} {v}" for k, v in lm.DEFAULT_CONF.items())
                    + f", yolo:* {lm.YOLO_CONF}")
    ap.add_argument("--folder", type=Path, help="label this folder without a display")
    ap.add_argument("--out", type=Path, help="result folder of --folder")
    ap.add_argument("--resume", action="store_true", help="skip images already labelled")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--cfg", help="cfg of a yolo:ckpt: model")
    ap.add_argument("--dataset", help="class names of a yolo:ckpt: model")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)

    if args.conf is None:
        args.conf = lm.default_conf(args.model)
    if args.list:
        print_list(args.device)
        return
    try:
        if args.folder:
            if not args.out:
                ap.error("--folder needs --out")
            names, prompts = session_classes(args.classes, args.preset)
            files = lio.list_images(args.folder, args.recursive)
            if not files:
                sys.exit(f"no image in {args.folder}")
            lab = lm.get_labeller(args.model, args.device, args.cfg, args.dataset)
            t0 = time.time()
            gen = label_folder(lab, files, args.folder, args.out, names, prompts, args.conf,
                               args.resume)
            prov = None
            while True:
                try:
                    done, total, rel = next(gen)
                    print(f"\r{done}/{total} {rel}"[:100].ljust(100), end="", flush=True)
                except StopIteration as stop:
                    prov = stop.value
                    break
            print(f"\n{len(prov['images'])} images labelled in {args.out} "
                  f"({time.time() - t0:.0f} s); unreadable: {len(prov['unreadable'])}")
            return
        missing = [m for m in missing_extras() if m.startswith("gradio")]
        if missing:
            sys.exit(f"{', '.join(missing)} missing: {lm.EXTRA_HINT}")
        if not args.model.startswith("yolo:"):
            lm.need_torch()
        build_app(args).launch(server_name="127.0.0.1", server_port=args.port, share=False)
    except (lm.MissingExtra, ValueError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
