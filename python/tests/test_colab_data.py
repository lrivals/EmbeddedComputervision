"""M14 (T14.1) : jeux archivés pour Colab (get_datasets.sh pack/unpack, notebooks/colab.py)."""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "get_datasets.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("tar") is None,
                                reason="bash et tar requis")


def _load_colab():
    spec = importlib.util.spec_from_file_location("colab",
                                                  ROOT / "tools" / "notebooks" / "colab.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sh(*args, data_dir, archive_dir=None):
    env = dict(os.environ, DATA_DIR=str(data_dir))
    if archive_dir:
        env["ARCHIVE_DIR"] = str(archive_dir)
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True,
                          text=True)


def _kaggle_exdark(data):
    """Arborescence de la version Kaggle d'ExDark, annotations GitHub à côté."""
    d = data / "ExDark Dataset"
    (d / "Bicycle").mkdir(parents=True)
    (d / "Bicycle" / "a.jpg").write_bytes(b"jpg")
    (d / "ExDark_Annno" / "ExDark_Annno" / "Bicycle").mkdir(parents=True)
    (d / "ExDark_Annno" / "ExDark_Annno" / "Bicycle" / "a.jpg.txt").write_text("% bbGt\n")
    (d / "imageclasslist.txt").write_text("a.jpg 1 1 1 1\n")
    (data / "exdark").mkdir()
    (data / "exdark" / "dl.zip.part").write_bytes(b"partiel")


def test_pack_unpack_roundtrip(tmp_path):
    src, arch, dst = tmp_path / "src", tmp_path / "arch", tmp_path / "dst"
    _kaggle_exdark(src)
    assert _sh("kaggle", data_dir=src).returncode == 0
    assert _sh("pack", "exdark", data_dir=src, archive_dir=arch).returncode == 0
    names = subprocess.run(["tar", "-tf", arch / "exdark.tar"], capture_output=True,
                           text=True).stdout.split()
    assert "exdark/imageclasslist.txt" in names
    assert not any(n.endswith(".part") for n in names)

    assert _sh("ready", "exdark", data_dir=dst).returncode == 1
    assert _sh("unpack", str(arch), "exdark", data_dir=dst).returncode == 0
    # Liens suivis à l'archivage : des fichiers, plus des liens vers les dossiers Kaggle.
    ann = dst / "exdark" / "ExDark_Annno" / "Bicycle" / "a.jpg.txt"
    assert ann.is_file() and not (dst / "exdark" / "ExDark_Annno").is_symlink()
    assert _sh("ready", "exdark", data_dir=dst).returncode == 0
    assert "déjà prêt" in _sh("unpack", str(arch), "exdark", data_dir=dst).stdout


def test_pack_absent_and_unknown(tmp_path):
    assert _sh("pack", "kitti", data_dir=tmp_path).returncode == 1
    assert _sh("unpack", str(tmp_path), "kitti", data_dir=tmp_path).returncode == 1
    assert _sh("ready", "foo", data_dir=tmp_path).returncode == 1


def test_prepare_data_from_drive(tmp_path):
    colab = _load_colab()
    src, drive, dst = tmp_path / "src", tmp_path / "drive", tmp_path / "dst"
    _kaggle_exdark(src)
    _sh("kaggle", data_dir=src)
    assert _sh("pack", "exdark", data_dir=src, archive_dir=drive / "data").returncode == 0

    assert colab.prepare_data("exdark", drive, data_dir=dst, download=False) == "drive"
    assert colab.prepare_data("exdark", drive, data_dir=dst, download=False) == "ready"
    with pytest.raises(RuntimeError, match="pack flir"):
        colab.prepare_data("flir", drive, data_dir=dst, download=False)
