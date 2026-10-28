"""M18, M21, M32, M38 : prétraitements sur disque de tools/prep_datasets.py (DroneVehicle,
UAVDT, ASL, Oxford-IIIT Pet, TrashNet, PREWID)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools import prep_datasets as P  # noqa: E402
from yolo.data.datasets import CONSTRUCTION_PPE_CLASSES as D_CPPE  # noqa: E402


def test_obb_to_yolo_border():
    # Losange de coins (200, 150), (300, 200), (200, 250), (100, 200) sur 840×712 : boîte
    # englobante [100, 300] × [150, 250], soit [0, 200] × [50, 150] une fois le cadre ôté.
    pts = [(200, 150), (300, 200), (200, 250), (100, 200)]
    line = "0 " + " ".join(f"{x / 840} {y / 712}" for x, y in pts)
    c, cx, cy, w, h = P.obb_to_yolo(line, 840, 712, 100).split()
    assert c == "0"
    np.testing.assert_allclose([float(v) for v in (cx, cy, w, h)],
                               [100 / 640, 100 / 512, 200 / 640, 100 / 512], atol=1e-6)
    # Entièrement dans le cadre : rien.
    assert P.obb_to_yolo("1 " + " ".join(["0.05"] * 8), 840, 712, 100) is None


def test_prep_dronevehicle(tmp_path):
    from PIL import Image

    src = tmp_path / "src"
    for sub in ("images", "labels"):
        (src / "train" / sub).mkdir(parents=True)
    for split in ("val", "test"):
        for sub in ("images", "labels"):
            (src / split / sub).mkdir(parents=True)
    Image.new("RGB", (840, 712), "white").save(src / "train" / "images" / "00007_jpg.rf.ab.jpg")
    (src / "train" / "labels" / "00007_jpg.rf.ab.txt").write_text(
        "0 0.2 0.2 0.3 0.2 0.3 0.3 0.2 0.3\n")
    out = tmp_path / "out"
    P.prep_dronevehicle(src, out, jobs=1)
    with Image.open(out / "train" / "images" / "00007.jpg") as img:
        assert img.size == (640, 512)
    assert (out / "train" / "labels" / "00007.txt").read_text().startswith("0 ")


def test_prep_uavdt(tmp_path):
    src = tmp_path / "src"
    for split in ("train", "test"):
        (src / split / "ann").mkdir(parents=True)
        (src / split / "img").mkdir(parents=True)
    ann = {"tags": [{"name": "sequence", "value": "M0203"}, {"name": "night"}],
           "size": {"width": 1024, "height": 540},
           "objects": [{"classTitle": "car", "points": {"exterior": [[10, 5], [29, 24]]},
                        "tags": [{"name": "small occlusion"}, {"name": "no out"}]}]}
    for name in ("M0203_img000001.jpg", "S0101_img000001.jpg"):
        (src / "test" / "img" / name).write_bytes(b"x")
        sot = dict(ann, objects=[dict(ann["objects"][0], classTitle="vehicle")])
        (src / "test" / "ann" / f"{name}.json").write_text(
            json.dumps(ann if name[0] == "M" else sot))
    out = tmp_path / "out"
    P.prep_uavdt(src, out)
    (r,) = json.loads((out / "annotations_test.json").read_text())  # séquence S écartée
    assert r["boxes"] == [[10, 5, 30, 25]] and r["labels"] == ["car"]
    assert r["sequence"] == "M0203" and r["tags"] == ["night"]
    assert r["occlusion"] == ["small"] and r["out"] == ["no"]
    assert (out / "images" / "test" / "M0203_img000001.jpg").exists()
    assert json.loads((out / "annotations_train.json").read_text()) == []


def test_landmarks_to_box():
    pts = [(50, 40), (150, 140), (100, 90)]
    np.testing.assert_allclose(P.landmarks_to_box(pts, 200, 200, margin=0.1),
                               [40, 30, 160, 150])
    # Clipped to the image.
    assert P.landmarks_to_box([(5, 0), (195, 200)], 200, 200) == [0.0, 0.0, 200.0, 200.0]


def test_asl_split():
    assert P.asl_frame("A1") == 1 and P.asl_frame("space3000") == 3000
    assert P.asl_frame("A_test") is None
    assert P.asl_split("del2550") == "train" and P.asl_split("del2551") == "val"


def test_asl_check():
    hand = [(50, 60), (100, 160)]
    assert P.asl_check(hand, 200, 200) is None
    assert P.asl_check([(190, 10), (230, 30), (240, 40)], 200, 200) == "outside"
    assert P.asl_check([(5, 5), (20, 40)], 200, 200) == "small"


def _fake_hands():
    """Fake detector: a hand everywhere except in image n° 3 (left out) ; one in `nothing`."""
    def detect(path):
        if path.stem.endswith("3") and not path.stem.startswith("nothing"):
            return None
        return np.array([[50.0, 60.0], [100.0, 160.0]]), "Right", 0.9, 0
    return detect


def test_prep_asl(tmp_path):
    from PIL import Image

    src = tmp_path / "ASL Alphabet"
    train = src / "asl_alphabet_train" / "asl_alphabet_train"
    test = src / "asl_alphabet_test" / "asl_alphabet_test"
    names = [train / c / f"{c}{n}.jpg" for c in ("A", "nothing") for n in (1, 3, 2551)]
    for p in names + [test / "A_test.jpg"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (200, 200)).save(p)
    out = tmp_path / "asl"
    P.prep_asl(src, out, _fake_hands, jobs=1)
    tr = json.loads((out / "annotations_train.json").read_text())
    assert [e["image"] for e in tr] == ["images/train/A1.jpg", "images/train/nothing1.jpg",
                                        "images/train/nothing3.jpg"]
    a1 = tr[0]
    assert a1["labels"] == ["A"] and a1["frame"] == 1 and a1["hand"] == "Right"
    np.testing.assert_allclose(a1["boxes"][0], [42.5, 45, 107.5, 175])
    assert tr[1]["boxes"] == [] and tr[1]["hand"] == "Right"  # negative, hand counted
    val = json.loads((out / "annotations_val.json").read_text())
    assert [e["class"] for e in val] == ["A", "nothing"]
    (t,) = json.loads((out / "annotations_test.json").read_text())
    assert t["class"] == "A" and t["frame"] is None
    assert json.loads((out / "rejected.json").read_text()) == [
        {"image": "images/train/A3.jpg", "class": "A", "split": "train", "reason": "no hand"}]
    assert (out / "images" / "val" / "A2551.jpg").stat().st_ino == names[2].stat().st_ino
    # Detections cached: the detector is not called again, the annotations are rebuilt.
    (out / "annotations_train.json").unlink()
    P.prep_asl(src, out, lambda: None, jobs=1)
    assert json.loads((out / "annotations_train.json").read_text()) == tr
    # Loader on the output.
    from yolo.data import datasets as D

    assert len(D.load("asl", out, "train")) == 3
    P.asl_preview(out, tmp_path / "preview.png", per_class=1)
    assert (tmp_path / "preview.png").exists()


# ---------------------------------------------------------------- M32: Oxford-IIIT Pet

def _trimap(h=100, w=200, parts=()):
    t = np.full((h, w), 2, np.uint8)
    for y0, y1, x0, x1 in parts:
        t[y0:y1, x0:x1] = 1
        t[y0, x0:x1] = 3  # boundary pixels belong to the mask
    return t


def test_trimap_box():
    # One animal.
    box, parts, share = P.trimap_box(_trimap(parts=[(10, 50, 20, 80)]))
    assert box == [20, 10, 80, 50] and parts == 1 and share == 40 * 60 / 20000
    # Two parts apart (one animal cut by a coat, or two animals): one box over both.
    box, parts, _ = P.trimap_box(_trimap(parts=[(10, 50, 20, 80), (60, 90, 120, 190)]))
    assert box == [20, 10, 190, 90] and parts == 2
    # Two touching: one component.
    box, parts, _ = P.trimap_box(_trimap(parts=[(10, 50, 20, 80), (10, 50, 80, 150)]))
    assert box == [20, 10, 150, 50] and parts == 1
    # A speck under PETS_SPECK of the image is ignored, a small part is in the box but not
    # counted.
    box, parts, _ = P.trimap_box(_trimap(parts=[(10, 50, 20, 80), (95, 97, 195, 197),
                                                (80, 90, 150, 160)]))
    assert box == [20, 10, 160, 90] and parts == 1
    assert P.trimap_box(_trimap())[0] is None


def test_pets_check():
    body = [20, 10, 80, 50]
    assert P.pets_check(body, 0.5, None) is None
    assert P.pets_check(body, 0.5, [30, 10, 60, 30]) is None
    assert P.pets_check(body, 0.5, [70, 10, 100, 30]) == "head outside"
    assert P.pets_check(body, 0.01, None) == "small"
    assert P.pets_check(None, 0.5, None) == "small"


def test_stratified_split():
    items = [f"{c}_{i}" for c in "ab" for i in range(10)]
    split = P.stratified_split(items, lambda s: s[0], (0.8, 0.2), ("train", "val"))
    assert sum(v == "val" for k, v in split.items() if k[0] == "a") == 2
    assert split == P.stratified_split(items[::-1], lambda s: s[0], (0.8, 0.2),
                                       ("train", "val"))  # fixed seed, order free


def test_prep_pets(tmp_path):
    from PIL import Image

    src = tmp_path / "Oxford-IIIT Pet"
    ann = src / "annotations"
    (ann / "trimaps").mkdir(parents=True)
    (ann / "xmls").mkdir()
    (src / "images").mkdir()
    stems = [f"Abyssinian_{i}" for i in range(1, 11)] + ["pug_1", "pug_2"]
    for st in stems:
        Image.new("RGB", (200, 100)).save(src / "images" / f"{st}.jpg")
        parts = [(10, 50, 20, 80)] if st != "pug_2" else [(10, 12, 20, 22)]  # pug_2: small
        Image.fromarray(_trimap(parts=parts)).save(ann / "trimaps" / f"{st}.png")
    (ann / "xmls" / "Abyssinian_1.xml").write_text(
        "<annotation><object><bndbox><xmin>31</xmin><ymin>11</ymin><xmax>60</xmax>"
        "<ymax>30</ymax></bndbox></object></annotation>")
    (ann / "trainval.txt").write_text("".join(f"{s} 1 1 1\n" for s in stems[:10]))
    (ann / "test.txt").write_text("pug_1 26 2 15\npug_2 26 2 15\n")
    out = tmp_path / "pets"
    P.prep_pets(src, out, jobs=1)
    tr = json.loads((out / "annotations_train.json").read_text())
    val = json.loads((out / "annotations_val.json").read_text())
    assert len(tr) == 9 and len(val) == 1
    (t,) = json.loads((out / "annotations_test.json").read_text())
    assert t["class"] == "pug" and t["species"] == "dog" and t["boxes"] == [[20, 10, 80, 50]]
    a1 = next(e for e in tr + val if e["image"].endswith("Abyssinian_1.jpg"))
    assert a1["head"] == [30, 10, 60, 30] and a1["species"] == "cat"
    assert json.loads((out / "rejected.json").read_text()) == [
        {"image": "images/test/pug_2.jpg", "class": "pug", "split": "test", "reason": "small"}]
    from yolo.data import datasets as D

    (s,) = D.load("pets", out)
    assert s["labels"].tolist() == [D.PETS_CLASSES.index("pug")]
    P.pets_preview(out, tmp_path / "preview.png")
    assert (tmp_path / "preview.png").exists()


# ---------------------------------------------------------------- M32: TrashNet

def _fake_trash():
    import autolabel_models as lm

    class Fake(lm.Labeller):
        """Glass: a box over the whole photo; trash: nothing; else a centred box (prompt of
        the class) and a weaker one (generic prompt)."""
        calls = 0

        def _predict(self, img, names, prompts, conf):
            Fake.calls += 1
            if names[0] == "trash":
                return np.zeros((0, 4)), np.zeros(0), np.zeros(0, np.int64)
            if names[0] == "glass":
                return np.array([[0.5, 0.5, 1.0, 1.0]]), np.array([0.6]), np.array([1])
            return (np.array([[0.5, 0.5, 0.5, 0.5], [0.4, 0.4, 0.2, 0.2]]),
                    np.array([0.8, 0.5]), np.array([0, 1]))
    return Fake


def test_prep_trashnet(tmp_path):
    from PIL import Image

    src = tmp_path / "TrashNet"
    for k, c in enumerate(("glass", "metal", "trash")):
        (src / "dataset-resized" / c).mkdir(parents=True)
        for i in range(1, 8):  # distinct bytes: the cache is keyed by SHA-1
            Image.new("RGB", (64, 48), (20 * i, 100 * k, 0)).save(
                src / "dataset-resized" / c / f"{c}{i}.jpg")
    out = tmp_path / "trashnet"
    fake = _fake_trash()
    P.prep_trashnet(src, out, fake, jobs=1)
    recs = {s: json.loads((out / f"annotations_{s}.json").read_text())
            for s in ("train", "val", "test")}
    assert [len(recs[s]) for s in ("train", "val", "test")] == [10, 2, 2]  # glass, metal
    glass = next(e for e in recs["train"] if e["class"] == "glass")
    np.testing.assert_allclose(glass["boxes"][0], [0, 0, 64, 48])  # full frame: kept
    e = next(e for e in recs["train"] if e["class"] == "metal")
    assert e["class"] == "metal" and e["labels"] == ["metal"] and e["score"] == 0.8
    np.testing.assert_allclose(e["boxes"][0], [16, 12, 48, 36])
    why = {r["reason"] for r in json.loads((out / "rejected.json").read_text())}
    assert why == {"no box"}
    prov = json.loads((out / "autolabel.json").read_text())
    assert len(prov["images"]) == 21 and prov["prompts"][-1] == "object"
    rec = next(r for r in prov["images"].values() if len(r["raw"]["labels"]) == 2)
    assert rec["raw"]["labels"] == [P.TRASHNET_CLASSES.index("metal"), 6]
    # Second run: the cache is read, the labeller never built.
    n = fake.calls
    P.prep_trashnet(src, out, lambda: None, jobs=1)
    assert fake.calls == n
    assert json.loads((out / "annotations_train.json").read_text()) == recs["train"]
    from yolo.data import datasets as D

    assert len(D.load("trashnet", out)) == 2
    P.trashnet_preview(out, tmp_path / "preview.png")
    # Hand check: the corrected label of a train image gives its IoU.
    P.trashnet_handcheck(out, tmp_path / "check", n=2)  # one per class
    assert len(list((tmp_path / "check").iterdir())) == 2
    lab = tmp_path / "result" / "labels"
    lab.mkdir(parents=True)
    (lab / f"{Path(e['image']).stem}.txt").write_text("1 0.5 0.5 0.5 0.25\n")
    ious = P.trashnet_handcheck_iou(out, tmp_path / "result")
    np.testing.assert_allclose(ious, [0.5])
    (ref,) = json.loads((out / "handcheck_val.json").read_text())
    assert ref["labels"] == ["metal"]


# ---------------------------------------------------------------- M38: PREWID

def test_prewid_frame_and_split():
    assert P.prewid_frame("210430_CAM0") == 210430 and P.prewid_frame("CAMLAB01_0126") == 126
    frames = list(range(100))
    s = P.time_split(frames)
    assert [sum(v == n for v in s.values()) for n in ("train", "val", "test")] == [70, 15, 15]
    assert max(f for f in frames if s[f] == "train") < min(f for f in frames if s[f] == "val")
    # snap: a cut never splits a run of consecutive numbers
    s = P.time_split(list(range(6)) + [10, 11, 12] + [20, 21], snap=True)  # cut at 8 → 9
    assert {s[f] for f in (10, 11, 12)} == {"train"} and {s[20], s[21]} == {"test"}
    np.testing.assert_allclose(P.yolo_corners((0, 0.5, 0.5, 0.25, 0.5), 640, 480),
                               [240, 120, 400, 360])
    assert P.yolo_corners((0, 0.05, 0.5, 0.2, 0.2), 100, 100)[0] == 0  # clipped


def test_prep_prewid(tmp_path):
    from PIL import Image

    src = tmp_path / "PREWID"
    img, gt = src / "01_Images", src / "02_GroundTruth"
    csv = ["filename;class_id;x_center;y_center;width;height"]
    for cam, n, size in (("CAM0", 20, (64, 48)), ("CAM2", 20, (64, 48)),
                         ("CAMLAB01", 4, (96, 54))):
        (img / cam).mkdir(parents=True)
        (gt / f"GT_{cam}").mkdir(parents=True)
        # plant: 10 consecutive empty frames, then 10 workers apart (gaps in the numbers)
        frames = range(n) if cam == "CAMLAB01" else list(range(10)) + list(range(20, 40, 2))
        for k, i in enumerate(frames):
            stem = f"CAMLAB01_{i:04d}" if cam == "CAMLAB01" else f"2100{i:02d}_{cam}"
            Image.new("RGB", size).save(img / cam / f"{stem}.jpg")
            rows = [] if k < n // 2 else ["0 0.5 0.5 0.25 0.5"]  # empty frames first
            (gt / f"GT_{cam}" / f"{stem}.txt").write_text("".join(r + "\n" for r in rows))
            csv += [f"{stem}.txt;" + r.replace(" ", ";") for r in rows]
    csv[-1] = csv[-1].replace("0.25", "0.3")  # a lab row that disagrees with its txt
    (gt / "groundtruth.csv").write_text("\n".join(csv) + "\n")
    out = tmp_path / "prewid"
    records, issues = P.prep_prewid(src, out)
    assert issues["csv"] == ["CAMLAB01_0003"] and issues["small"] == []
    assert sum(len(r) for r in records.values()) == 44 and len(records["lab"]) == 4
    for sp in ("train", "val", "test"):  # empty frames and workers in every split
        es = json.loads((out / f"annotations_{sp}.json").read_text())
        assert {e["camera"] for e in es} == {"CAM0", "CAM2"}
        assert any(not e["boxes"] for e in es) and any(e["boxes"] for e in es)
    e = next(e for e in records["train"] if e["boxes"])
    assert e["boxes"] == [[24, 12, 40, 36]] and e["labels"] == ["person"]
    assert (out / e["image"]).stat().st_nlink == 2  # hard link to the raw frame
    loco = json.loads((out / "annotations_loco_CAM2_test.json").read_text())
    assert len(loco) == 20 and {e["camera"] for e in loco} == {"CAM2"}
    assert {e["camera"] for e in json.loads(
        (out / "annotations_loco_CAM2_train.json").read_text())} == {"CAM0"}
    from yolo.data import datasets as D

    assert len(D.load("prewid", out, "test:CAM0")) == len(
        [e for e in records["test"] if e["camera"] == "CAM0"])
    assert len(D.load("prewid", out, "lab")) == 4
    P.prewid_preview(out, tmp_path / "preview.png")
    assert (tmp_path / "preview.png").exists()


def test_prep_construction_ppe(tmp_path):
    from PIL import Image

    src = tmp_path / "construction-ppe"
    person = D_CPPE.index("Person")
    for sp, rows in (("train", [f"{person} 0.5 0.5 0.25 0.5", "0 0.5 0.2 0.1 0.1"]),
                     ("val", ["7 0.5 0.2 0.1 0.1"]), ("test", [])):
        (src / "images" / sp).mkdir(parents=True)
        (src / "labels" / sp).mkdir(parents=True)
        Image.new("RGB", (64, 48)).save(src / "images" / sp / f"{sp}_1.jpg")
        (src / "labels" / sp / f"{sp}_1.txt").write_text("".join(r + "\n" for r in rows))
    out = tmp_path / "construction_ppe"
    ppe = P.prep_construction_ppe(src, out)
    assert ppe["train"][0]["labels"] == ["Person", "helmet"]
    (tr,) = json.loads((out / "annotations_train.json").read_text())
    assert tr["boxes"] == [[24, 12, 40, 36]] and tr["labels"] == ["person"]
    (val,) = json.loads((out / "annotations_val.json").read_text())
    assert val["boxes"] == [] and val["labels"] == []  # no_helmet only: no person
    assert json.loads((out / "annotations_ppe_val.json").read_text())[0]["labels"] == [
        "no_helmet"]
    assert (out / tr["image"]).stat().st_nlink == 2
    from yolo.data import datasets as D

    (s,) = D.load("construction_ppe", out, "train")
    assert s["labels"].tolist() == [0]
