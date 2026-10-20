"""T28.0: the equivalence check reports a forged gap; the matrix section is regenerated as is."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("equivalence", ROOT / "tools" / "equivalence.py")
eq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eq)

NET = "tiny-yolov2-voc"
needs_model = pytest.mark.skipif(not (ROOT / "model" / NET / "manifest.json").exists(),
                                 reason="export absent (make export)")


@needs_model
def test_committed_dumps_reproduce():
    assert eq.check_dumps([NET]) == []


@needs_model
def test_forged_byte_reported(tmp_path):
    src = ROOT / "model" / NET
    dst = tmp_path / NET
    dst.mkdir()
    for f in src.iterdir():
        if f.name != "dumps":
            (dst / f.name).symlink_to(f)
    img = sorted((src / "dumps").iterdir())[0]
    shutil.copytree(img, dst / "dumps" / img.name)
    a = np.load(dst / "dumps" / img.name / "L05.npy")
    a.flat[123] ^= 1
    np.save(dst / "dumps" / img.name / "L05.npy", a)
    bad = eq.check_dumps([NET], tmp_path)
    assert bad == [f"{NET} {img.name} L05: 1 values"]


@pytest.mark.skipif(not eq.DATA.exists(), reason="results/equivalence.json absent")
def test_report_idempotent():
    data = json.loads(eq.DATA.read_text())
    doc = eq.DOC.read_text()
    once = eq.splice(doc, eq.render(data))
    assert eq.splice(once, eq.render(data)) == once
    assert once.count(eq.BEGIN) == 1
