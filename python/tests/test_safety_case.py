"""M31: the evidence leaves of the assurance argument point to existing files (T31.4)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("safety_case", ROOT / "tools" / "safety_case.py")
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)

ARG = """# A

```mermaid
graph TD
  G0["G0: goal"] --> S1["S1: one"]
  G0 --> S2["S2: two"]
```

| Leaf | Evidence | Path | Supports |
|---|---|---|---|
| S1 | one | `results/a.md` | met |
| S2 | two | `results/b.md` | met |

See [odd](odd.md) and [the web](https://example.org) and [here](#a).
"""


def _repo(tmp_path, text=ARG):
    (tmp_path / "docs" / "safety").mkdir(parents=True)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "a.md").write_text("")
    (tmp_path / "results" / "b.md").write_text("")
    (tmp_path / "docs" / "safety" / "odd.md").write_text("")
    (tmp_path / "docs" / "safety" / "argument.md").write_text(text)
    return tmp_path


def test_repo_case_consistent():
    assert sc.check() == []


def test_fake_repo_passes(tmp_path):
    assert sc.check(_repo(tmp_path)) == []


def test_missing_path_detected(tmp_path):
    root = _repo(tmp_path)
    (root / "results" / "b.md").unlink()
    assert sc.check(root) == ["argument.md: S2 → results/b.md does not exist"]


def test_leaf_without_row_and_row_without_leaf(tmp_path):
    text = ARG.replace('G0 --> S2["S2: two"]', 'G0 --> S3["S3: three"]')
    problems = sc.check(_repo(tmp_path, text))
    assert "argument.md: leaf S3 has no row in the evidence table" in problems
    assert "argument.md: row S2 is not a leaf of the tree" in problems


def test_broken_link_detected(tmp_path):
    root = _repo(tmp_path)
    (root / "docs" / "safety" / "odd.md").write_text("[r](../../results/none.md)")
    assert sc.check(root) == ["docs/safety/odd.md: link ../../results/none.md does not exist"]
