"""T0.4 : le manifest d'exemple décrit les 24 couches du §3.2 et respecte le schéma."""

import importlib.util
import json
from pathlib import Path

import jsonschema
import pytest

from yolo.models.specs import TINY_YOLOV3_VOC, infer_shapes

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "docs" / "manifest.schema.json").read_text())
MANIFEST_PATH = ROOT / "model" / "example_manifest.json"


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def layers(manifest):
    return manifest["layers"]


def nbytes(shape):
    c, h, w = shape
    return c * h * w


def test_schema_is_valid():
    jsonschema.Draft202012Validator.check_schema(SCHEMA)


def test_manifest_matches_schema(manifest):
    jsonschema.Draft202012Validator(SCHEMA).validate(manifest)


def test_schema_rejects_unknown_layer(manifest):
    bad = json.loads(json.dumps(manifest))
    bad["layers"][0]["type"] = "dense"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(SCHEMA).validate(bad)


def test_generator_is_up_to_date():
    spec = importlib.util.spec_from_file_location(
        "make_example_manifest", ROOT / "tools" / "make_example_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert MANIFEST_PATH.read_text() == mod.dumps(mod.build())


def test_24_layers_of_section_3_2(layers):
    specs = TINY_YOLOV3_VOC["layers"]
    shapes = infer_shapes(TINY_YOLOV3_VOC)
    assert [e["id"] for e in layers] == list(range(24))
    assert [e["type"] for e in layers] == [s["type"] for s in specs]
    for e, s, (ins, outs) in zip(layers, specs, shapes):
        if e["type"] == "conv":
            assert (e["k"], e["cin"], e["cout"], e["act"]) == (s["k"], ins[0], s["cout"], s["act"])
            if e["fused_pool"] is None:
                assert tuple(e["out_shape"]) == outs
        else:
            assert tuple(e["out_shape"]) == outs
        if e["type"] == "route":
            assert e["from"] == s["from"]


def test_maxpools_are_fused(layers):
    for e in layers:
        if e["type"] != "maxpool":
            continue
        conv = layers[e["fused_into"]]
        assert conv["type"] == "conv"
        assert conv["fused_pool"] == {"layer": e["id"], "k": e["k"], "s": e["s"]}
        assert conv["out_shape"] == e["out_shape"]


def test_route_20_contiguous(layers):
    # §10.3 : la couche 19 puis la couche 8 (avant pooling) bout à bout, même échelle.
    route = layers[20]
    l19, l8 = layers[19], layers[8]
    assert route["from"] == [19, 8]
    assert route["out"] == l19["out"]
    assert l8["prepool_out"]["buf"] == l19["out"]["buf"]
    assert l8["prepool_out"]["offset"] == l19["out"]["offset"] + nbytes(l19["out_shape"])
    assert l19["scale"] == l8["out_scale"] == route["scale"]
    assert layers[21]["in"] == route["out"]


def test_blob_offsets(manifest, layers):
    convs = [e for e in layers if e["type"] == "conv"]
    w_end = b_end = 0
    for e in convs:
        assert e["w_offset"] >= w_end and e["b_offset"] >= b_end and e["m0_offset"] == e["b_offset"]
        w_end = e["w_offset"] + e["k"] ** 2 * e["cin"] * e["cout"]
        b_end = e["b_offset"] + 4 * e["cout"]
    assert w_end <= manifest["blobs"]["weights.bin"]
    assert b_end <= manifest["blobs"]["bias.bin"] == manifest["blobs"]["requant.bin"]


def test_buffers_fit_and_never_clobber_live_tensors(manifest, layers):
    shapes = infer_shapes(TINY_YOLOV3_VOC)
    sizes = manifest["buffers"]

    # Région de chaque tenseur produit, et instant de sa dernière lecture.
    def producer(i):
        """Couche qui écrit physiquement la sortie logique de la couche i."""
        e = layers[i]
        return e["fused_into"] if e["type"] == "maxpool" else i

    region = {-1: (manifest["input"]["buf"]["buf"], 0, nbytes(manifest["input"]["shape"][1:]))}
    for e in layers:
        if e["type"] in ("conv", "upsample"):
            o = e["out"]
            region[e["id"]] = (o["buf"], o["offset"], o["offset"] + nbytes(e["out_shape"]))
        if "prepool_out" in e:
            p = e["prepool_out"]
            region[(e["id"], "pre")] = (p["buf"], p["offset"], p["offset"] + nbytes(shapes[e["id"]][1]))

    reads = {}  # tenseur → dernière couche qui le lit
    prev = -1
    for e in layers:
        i, t = e["id"], e["type"]
        if t == "route":
            srcs = [(j, "pre") if (j, "pre") in region else producer(j) for j in e["from"]]
            for s in srcs:
                reads[s] = i
            prev = srcs[0]  # la suite lit à l'adresse de la route
            continue
        if t in ("conv", "upsample", "yolo"):
            reads[prev] = i
        if t != "yolo":
            prev = producer(i)

    for name, (buf, start, end) in region.items():
        assert end <= sizes[buf], f"{name} dépasse le tampon {buf}"

    def written_at(name):
        return name[0] if isinstance(name, tuple) else name

    for a, (buf_a, s_a, e_a) in region.items():
        t = written_at(a)
        for b, (buf_b, s_b, e_b) in region.items():
            if a == b or buf_a != buf_b or not (s_a < e_b and s_b < e_a):
                continue
            # b vivant quand a est écrit : produit avant, lu par la même couche ou après.
            assert not (written_at(b) < t <= reads.get(b, -1)), f"{a} écrase {b}"
