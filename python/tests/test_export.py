"""T4.7 : export du modèle entier — manifest valide, blobs relus == modèle, dumps rejoués ;
T24.3 : YOLOv4-tiny (routes groupées, maxpool après concaténation) et test d'allocation ;
T26.0 : YOLOv7-tiny (concaténations de 4 sources, SPP réécrit en maxpools 5×5 chaînés)."""

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from yolo.io.export import build_manifest, export_model, layout, load_luts, load_model
from yolo.models.specs import NETWORKS, infer_shapes
from yolo.models.tiny_yolo import build, load_cfg
from yolo.quant.calibrate import ActStats, choose_scales
from yolo.quant.fuse_bn import engine_net, fuse_network
from yolo.quant.int_model import IntNetwork, QuantModel, head_luts
from yolo.quant.quantize import INPUT_SCALE, quantize_input

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "docs" / "manifest.schema.json").read_text())
EXPORTS = [ROOT / "model" / n for n in ("tiny-yolov2-voc", "tiny-yolov3-coco")]


def _qm(name, rng):
    fused = fuse_network(build(name, dtype=np.float64, rng=1))
    stats = ActStats(fused, per_image=1024, rng=0)
    x = rng.uniform(0, 1, (1, 3, 64, 64))
    stats.update(fused.forward(x, train=False, all_outputs=True))
    scales, _, _ = choose_scales(fused, stats)
    return QuantModel.from_fused(fused, INPUT_SCALE, scales), quantize_input(x)


@pytest.fixture(scope="module", params=["tiny-yolov2-voc", "tiny-yolov3-voc",
                                        "tiny-yolov4-coco", "tiny-yolov7-coco"])
def exported(request, tmp_path_factory):
    qm, qx = _qm(request.param, np.random.default_rng(0))
    d = tmp_path_factory.mktemp(request.param)
    return qm, qx, d, export_model(qm, d)


def test_manifest_valid(exported):
    _, _, d, m = exported
    jsonschema.Draft202012Validator(SCHEMA).validate(m)
    assert json.loads((d / "manifest.json").read_text()) == m
    for name, size in m["blobs"].items():
        assert (d / name).stat().st_size == size


def test_blobs_reloaded_identical(exported):
    qm, qx, d, _ = exported
    rq, m = load_model(d)
    assert rq.input_scale == qm.input_scale
    for i, c in qm.convs.items():
        r = rq.convs[i]
        assert np.array_equal(r.qW, c.qW) and np.array_equal(r.qb, c.qb)
        assert np.array_equal(r.M0, c.M0) and r.n == c.n
        assert (r.sx, r.sy, r.act) == (c.sx, c.sy, c.act)
    a = IntNetwork(qm).forward(qx, all_outputs=True)
    b = IntNetwork(rq).forward(qx, all_outputs=True)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    luts = load_luts(d, m)
    for hid, h in head_luts(qm).items():
        for name, table in luts[hid].items():
            assert np.array_equal(table, getattr(h, name))


def test_v2_region_and_float_anchors(exported):
    qm, _, _, m = exported
    if qm.net["name"] != "tiny-yolov2-voc":
        pytest.skip("YOLOv2 seulement")
    assert m["layers"][-1]["type"] == "region" and m["layers"][-1]["num"] == 5
    assert m["anchors"][0] == [pytest.approx(34.56), pytest.approx(38.08)]
    # Chaîne simple : ping-pong A/B, tête dans H13, aucun tampon spécial.
    assert set(m["buffers"]) == {"A", "B", "H13"}


def test_layout_rejects_unfused_maxpool():
    net = {"input": (3, 32, 32), "layers": [{"type": "maxpool", "k": 2, "s": 2}]}
    with pytest.raises(ValueError):
        layout(net)


def _tensors(m):
    """Tenseurs écrits en DDR, dans le modèle d'exécution du matériel : [(couche qui l'écrit,
    dernière couche qui le lit, [(tampon, début, fin)])]. Upsample et route sont des vues sans
    copie (§10.3) ; une conv à maxpool fusionné écrit deux tenseurs, sa carte poolée et, s'il
    le faut, sa carte avant pooling (`prepool_out`) ; les têtes vivent jusqu'à la fin.
    """
    layers = m["layers"]
    n = len(layers)
    cons = {i: [] for i in range(-1, n)}
    for e in layers:
        for j in (e["from"] if e["type"] == "route" else [e["id"] - 1]):
            cons[j].append(e["id"])

    def eff(c):  # dernière lecture « réelle » à travers une vue
        t = layers[c]["type"]
        if t == "conv":
            return c
        if t in ("yolo", "region"):
            return n
        return max([eff(d) for d in cons[c]], default=c)

    def span(ref, size):
        return [(ref["buf"], ref["offset"], ref["offset"] + size)]

    out = [(-1, max(eff(c) for c in cons[-1]), span(m["input"]["buf"],
                                                      int(np.prod(m["input"]["shape"]))))]
    for e in layers:
        if e["type"] != "conv":
            continue
        i = e["id"]
        pool = e["fused_pool"]["layer"] if e["fused_pool"] else None
        if pool is None:
            out.append((i, max(eff(c) for c in cons[i]), span(e["out"],
                                                            int(np.prod(e["out_shape"])))))
            continue
        out.append((i, eff(pool), span(e["out"], int(np.prod(e["out_shape"])))))
        via = layers[pool]["fused_into"]  # la conv elle-même, ou la route poolée
        others = [c for c in cons[i] if c not in (pool, via)]
        if "prepool_out" in e:
            _, h, w = e["out_shape"]
            f = e["fused_pool"]["s"]
            out.append((i, max([eff(c) for c in others], default=i),
                        span(e["prepool_out"], e["cout"] * h * f * w * f)))
        else:
            assert not others, f"conv {i} : carte avant pooling lue mais jamais écrite"
    return out


def check_allocation(m):
    """Aucune écriture ne recouvre un tenseur encore à lire (ni l'entrée de la couche)."""
    tensors = _tensors(m)
    for i, _, places in tensors:
        for buf, a, b in places:
            for j, end, other in tensors:
                if j >= i or end < i:
                    continue
                for buf2, a2, b2 in other:
                    assert not (buf == buf2 and a < b2 and a2 < b), (
                        f"couche {i} écrit {buf}[{a}:{b}] sur la sortie de {j} ({buf2}[{a2}:{b2}],"
                        f" lue jusqu'à {end})")


@pytest.mark.parametrize("name", ["tiny-yolov2-voc", "tiny-yolov3-voc", "tiny-yolov4-coco",
                                  "tiny-yolov7-coco"])
def test_allocation_no_overlap(name):
    # Avec les ancres (la table de specs de v2 n'a pas de [region]) ; graphe du moteur.
    net = engine_net(load_cfg(name))
    m = build_manifest(net, 1.0 / 127, lambda i: 0.05, lambda i: 16)
    jsonschema.Draft202012Validator(SCHEMA).validate(m)
    check_allocation(m)


def test_allocation_check_detects_overlap():
    m = build_manifest(NETWORKS["tiny-yolov3-voc"], 1.0 / 127, lambda i: 0.05, lambda i: 16)
    m["layers"][14]["out"] = dict(m["layers"][13]["out"])  # écrit sur l'entrée de la conv 14
    with pytest.raises(AssertionError, match="couche 14"):
        check_allocation(m)


def test_v4_grouped_routes_and_pooled_concat():
    net = NETWORKS["tiny-yolov4-coco"]
    shapes = infer_shapes(net)
    m = build_manifest(net, 1.0 / 127, lambda i: 0.05, lambda i: 16)
    e = {x["id"]: x for x in m["layers"]}
    # Route groupée : la seconde moitié des canaux de sa source, sans copie.
    for r, src in ((3, 2), (11, 10), (19, 18)):
        c, h, w = shapes[src][1]
        assert e[r]["layout"] == "grouped" and (e[r]["groups"], e[r]["group_id"]) == (2, 1)
        assert e[src]["prepool_out"]["buf"] == e[r]["out"]["buf"]
        assert e[r]["out"]["offset"] == e[src]["prepool_out"]["offset"] + (c // 2) * h * w
        assert e[r + 1]["in"] == e[r]["out"]
    # Maxpool après concaténation : fusionné dans chaque source, écrites poolées bout à bout.
    for route, pool in ((8, 9), (16, 17), (24, 25)):
        assert e[pool]["fused_into"] == route
        a, b = e[route]["from"]
        assert e[a]["fused_pool"]["layer"] == e[b]["fused_pool"]["layer"] == pool
        assert e[a]["out"]["buf"] == e[b]["out"]["buf"] == e[route]["out"]["buf"]
        assert e[b]["out"]["offset"] == e[a]["out"]["offset"] + int(np.prod(e[a]["out_shape"]))
        assert e[a]["out_shape"][1:] == list(shapes[pool][1][1:])
        assert e[pool + 1]["in"] == e[route]["out"]
    # La conv 23 : poolée dans la route 24, avant pooling dans la route 34.
    assert e[23]["prepool_out"]["buf"] == e[34]["out"]["buf"]
    assert [e[i]["s"] for i in (0, 1, 2)] == [2, 2, 1]
    assert e[30]["scale_x_y"] == e[37]["scale_x_y"] == 1.05


def test_v7_spp_and_four_source_concats():
    net = engine_net(NETWORKS["tiny-yolov7-coco"])
    shapes = infer_shapes(net)
    m = build_manifest(net, 1.0 / 127, lambda i: 0.05, lambda i: 16)
    e = {x["id"]: x for x in m["layers"]}

    def at(ref):
        return ref["buf"], ref["offset"]

    # SPP : L35 écrit son pool 5×5 (L36) et sa carte (L35) ; les convs identité 37 et 39
    # écrivent pool5∘pool5 = pool9 (L38) et pool13 (L40) ; tout bout à bout dans la route 41.
    assert [e[i]["k"] for i in (36, 38, 40)] == [5, 5, 5]
    assert [e[i]["fused_into"] for i in (36, 38, 40)] == [35, 37, 39]
    assert e[37]["in"] == e[35]["out"] and e[39]["in"] == e[37]["out"]
    order = [e[39]["out"], e[37]["out"], e[35]["out"], e[35]["prepool_out"]]
    buf, off = at(e[41]["out"])
    for ref in order:
        assert at(ref) == (buf, off)
        off += 256 * 13 * 13
    # Concaténations de 4 convs : sources bout à bout, aucune copie.
    for r in (7, 15, 23, 31, 55, 67, 76, 85):
        buf, off = at(e[r]["out"])
        for j in net["layers"][r]["from"]:
            assert at(e[j]["out"]) == (buf, off), (r, j)
            off += int(np.prod(shapes[j][1]))
    assert all(e[h]["new_coords"] == 1 and e[h]["scale_x_y"] == 2.0 for h in (90, 94, 98))


def test_v7_lowering_keeps_outputs():
    net = build("tiny-yolov7-coco", dtype=np.float64, rng=2)
    fused = fuse_network(net)
    assert [fused.layers[i]["type"] for i in range(35, 42)] == [
        "conv", "maxpool", "conv", "maxpool", "conv", "maxpool", "route"]
    assert all(fused.layers[i]["k"] == 5 for i in (36, 38, 40))
    x = np.random.default_rng(3).uniform(0, 1, (1, 3, 64, 64))
    a = net.forward(x, train=False, all_outputs=True)
    b = fused.forward(x, train=False, all_outputs=True)
    for i in (36, 38, 40, 41, 98):
        np.testing.assert_allclose(b[i], a[i], rtol=1e-9, atol=1e-9)


def test_identity_conv_exact_in_integer():
    # La conv identité du SPP (W = I, b = 0, s_x = s_y) rend son entrée à l'entier près.
    from yolo.quant.int_layers import conv_int
    from yolo.quant.quantize import quantize_weights_per_channel, requant_params

    c, s = 4, 0.0492
    qW, sw = quantize_weights_per_channel(np.eye(c)[:, :, None, None])
    M0, n = requant_params(s, sw, s)
    q = np.arange(-127, 128).reshape(1, 1, 15, 17).repeat(c, axis=1)
    y, _ = conv_int(q, qW, np.zeros(c, np.int64), M0, n, act="linear")
    np.testing.assert_array_equal(y, q)


def test_v4_concat_sources_share_scale():
    from yolo.quant.calibrate import unify_routes

    layers = NETWORKS["tiny-yolov4-coco"]["layers"]
    convs = [i for i, x in enumerate(layers) if x["type"] == "conv"]
    scales, _ = unify_routes(layers, {i: 0.01 * (i + 1) for i in convs})
    for r in (6, 8, 14, 16, 22, 24):
        a, b = layers[r]["from"]
        assert scales[a] == scales[b], r
    assert scales[18] == scales[23] == scales[32]  # routes 24 et 34 (via l'upsample 33)


@pytest.mark.parametrize("d", EXPORTS, ids=lambda p: p.name)
def test_real_export_dumps_replay(d):
    """Export réel (`make export`) : manifest valide, dumps == passe avant du modèle relu."""
    if not (d / "manifest.json").exists():
        pytest.skip(f"{d.name} non exporté (make export)")
    qm, m = load_model(d)
    jsonschema.Draft202012Validator(SCHEMA).validate(m)
    dumps = sorted((d / "dumps").iterdir())
    assert len(dumps) == 3
    for dd in dumps[:1]:  # une image suffit ici ; les 3 sont rejouées par le golden C++
        qx = np.load(dd / "input.npy")
        outs = IntNetwork(qm).forward(qx, all_outputs=True)
        assert len(outs) == len(m["layers"])
        for i, o in enumerate(outs):
            ref = np.load(dd / f"L{i:02d}.npy")
            assert ref.dtype == np.int8 and np.array_equal(o, ref), i


def test_pack4_roundtrip():
    from yolo.io.export import pack4, unpack4, weight_bytes

    q = np.random.default_rng(0).integers(-8, 8, 37)
    b = pack4(q)
    assert b.dtype == np.int8 and len(b) == weight_bytes(37, 4) == 19
    assert np.array_equal(unpack4(b, 37), q)
    # quartet bas = poids d'indice pair
    assert pack4(np.array([1, -1])).view(np.uint8)[0] == 0xF1
    with pytest.raises(ValueError):
        pack4(np.array([8]))


def test_export_wbits4_roundtrip(tmp_path):
    qm, qx = _qm("tiny-yolov2-voc", np.random.default_rng(0))
    i = sorted(qm.convs)[1]
    c = qm.convs[i]
    c.qW = np.clip(c.qW, -7, 7).astype(np.int8)
    c.wbits = 4
    m = export_model(qm, tmp_path)
    e = next(e for e in m["layers"] if e["id"] == i)
    assert e["wbits"] == 4
    jsonschema.Draft202012Validator(SCHEMA).validate(m)
    back, _ = load_model(tmp_path)
    assert back.convs[i].wbits == 4 and np.array_equal(back.convs[i].qW, c.qW)
    a, b = IntNetwork(qm).forward(qx), IntNetwork(back).forward(qx)
    assert all(np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("d", EXPORTS, ids=lambda p: p.name)
def test_real_dump_inputs_reproduced(d):
    """Prétraitement actuel (letterbox PIL de `preprocess`, puis `quantize_input`) == entrées
    des dumps exportés, à l'octet : garde de non-régression du prétraitement (T11.4, T11.7)."""
    from PIL import Image

    from yolo.infer.pipeline import preprocess

    jpegs = ROOT / "data" / "VOCdevkit" / "VOC2007" / "JPEGImages"
    if not (d / "manifest.json").exists() or not jpegs.exists():
        pytest.skip(f"{d.name} non exporté ou VOC2007 absent")
    for dd in sorted((d / "dumps").iterdir()):
        with Image.open(jpegs / f"{dd.name}.jpg") as img:
            x, _ = preprocess(img, 416)
        assert np.array_equal(quantize_input(x[None]), np.load(dd / "input.npy")), dd.name
