"""M17: generated blocks of the article (tools/article.py) — parsing, rendering, --check,
collect from the sources."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import article as A  # noqa: E402


@dataclass
class Fig:
    name: str
    family: str
    caption: str
    subset: bool = False
    dest: Path | None = None

    @property
    def command(self):
        return f"python -m tools.figures {self.name}"


SUIVI = """# Plan

## Suivi

| Jalon | Tâches | Faites |
|---|---|---|
| M0 | 6 | 6 |
| M1 | 9 | 8 (T1.9 en cours) |

## Autre
"""

TEXT = """# Title

## 1. Introduction

Narrative with 3.14 outside results, kept as is.
State: <!-- article:etat:resume --><!-- /article -->.

## 4. Inference

The mAP is <!-- article:n:map.voc.int8 -->old<!-- /article --> points, threshold 0.45.

<!-- article:table:t -->
<!-- /article -->

<!-- article:fig:graphe/graphe_net -->
stale
<!-- /article -->

## 11. Limits

<!-- article:list:status --><!-- /article -->

<!-- article:etat --><!-- /article -->

Rendered at <!-- article:rev --><!-- /article -->.
"""

DATA = {
    "map.voc.int8": {"value": 55.66, "status": "measured", "nd": 2},
    "map.voc.csim": {"value": 55.66, "status": "csim", "nd": 2},
    "t": {"value": {"header": ["A", "B"], "rows": [["x", 1234.5], ["y", -0.5]],
                    "nd": [None, 1]}, "status": "projection"},
    "_rev": {"rev": "abc1234", "date": "2026-10-07"},
}


@pytest.fixture
def env(tmp_path):
    png = tmp_path / "results" / "figures" / "modeles" / "graphe_net.png"
    png.parent.mkdir(parents=True)
    png.write_bytes(b"png")
    (tmp_path / "suivi.md").write_text(SUIVI)
    article = tmp_path / "docs" / "article" / "article.md"
    article.parent.mkdir(parents=True)
    article.write_text(TEXT)
    figs = {"graphe": Fig("graphe", "modeles", "Graph of the net."),
            "sub": Fig("sub", "resultats", "Subset.", subset=True)}
    return {"root": tmp_path, "article": article, "figures": figs,
            "suivi_path": tmp_path / "suivi.md"}


def _render(text, env, data=DATA):
    return A.render(text, data, env["figures"], env["article"], env["root"], env["suivi_path"])


def _check(text, env, data=DATA):
    return A.check(text, data, env["figures"], env["article"], env["root"], env["suivi_path"])


def test_render_blocks(env):
    out = _render(TEXT, env)
    assert "<!-- article:n:map.voc.int8 -->55.66<!-- /article -->" in out
    assert "| x | 1,234.5 |" in out and "| y | −0.5 |" in out
    assert "![graphe_net](../../results/figures/modeles/graphe_net.png)" in out
    assert "*Figure 1 (`graphe`): Graph of the net.* <!-- python -m tools.figures graphe -->" in out
    assert "14 of 15 tasks done across 2 milestones, 1 milestones complete" in out
    assert "| M1 | 9 | 8 | 89 % |" in out
    assert "rev. `abc1234`, 2026-10-07" in out
    assert "- **csim** (1): `map.voc.csim`" in out
    assert "- **projection** (1): `t`" in out


def test_status_mark_and_format():
    assert A.render_n({"value": 37.2, "status": "projection", "nd": 1}) == "37.2<sup>proj.</sup>"
    assert A.render_n({"value": -0.64, "status": "measured", "nd": 2, "sign": True}) == "−0.64"
    assert A.fmt(0.1, 2, sign=True) == "+0.10"
    assert A.fmt(4952) == "4,952"
    assert A.fmt(-0.001, 2) == "0.00"


def test_idempotent_and_outside_untouched(env):
    once = _render(TEXT, env)
    assert _render(once, env) == once
    outside = [s.split("<!-- /article -->")[-1] for s in TEXT.split("<!-- article:")]
    outside_after = [s.split("<!-- /article -->")[-1] for s in once.split("<!-- article:")]
    assert outside == outside_after
    assert once.startswith(TEXT[:TEXT.index("<!-- article:")])


@pytest.mark.parametrize("bad, why", [
    ("x <!-- article:n:a --> y\n", "never closed"),
    ("<!-- article:n:a --> <!-- article:n:b --><!-- /article -->", "inside the block"),
    ("text\n<!-- /article -->", "without an opening"),
    ("\n\n<!-- article:foo:a --><!-- /article -->", "line 3: unknown block type"),
])
def test_malformed_blocks(bad, why):
    with pytest.raises(A.ArticleError, match=why):
        A.parse(bad)


def test_check(env):
    good = _render(TEXT, env)
    assert _check(good, env) == []
    assert _check(TEXT, env)  # not rendered yet
    hand = good.replace("-->55.66<!--", "-->55.70<!--")
    probs = _check(hand, env)
    assert len(probs) == 1 and "n:map.voc.int8" in probs[0] and "55.70" in probs[0]
    # a missing key, an unknown figure, a subset figure
    for block, msg in (("n:nope", "absent from chiffres.json"),
                       ("fig:inconnue", "absent from the registry"),
                       ("fig:sub", "not published"),
                       ("fig:graphe/absent", "absent")):
        probs = _check(good + f"\n<!-- article:{block} --><!-- /article -->\n", env)
        assert any(msg in p for p in probs), (block, probs)
    # second figure is numbered 2
    two = _render(good + "\n<!-- article:fig:graphe/graphe_net --><!-- /article -->\n", env)
    assert "*Figure 2 (`graphe`)" in two and _check(two, env) == []


def test_stray_numbers(env):
    good = _render(TEXT, env)
    assert A.stray_numbers(good) == []  # 3.14 in §1, 0.45 whitelisted
    bad = good.replace("threshold 0.45.", "threshold 0.45, gain 1.37 [2024-zhang#013.1], "
                       "`p99.99`, $1.5$, §10.4, T17.2.")
    found = A.stray_numbers(bad)
    assert [n for _, n in found] == ["1.37"]
    assert any("outside a block" in p for p in _check(bad, env))


def test_md_helpers(tmp_path):
    p = tmp_path / "r.md"
    p.write_text("# T\n\n| Stade | mAP |\n|---|---|\n| a | **56,30** |\n| b | −0,64 point |\n")
    header, rows = A.table_with(p, "Stade", "mAP")
    assert header == ["Stade", "mAP"] and [A.num(r[1]) for r in rows] == [56.30, -0.64]
    assert A.num("4 952 images") == 4952 and A.num("—") is None
    with pytest.raises(A.MissingSource):
        A.table_with(p, "absent")


def test_collect(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("1.5")
    other = tmp_path / "other.txt"
    other.write_text("7")
    keys = {"a": A.Key(lambda r: float(A.need(src).read_text()), "src.txt", "cmd", "measured", 1),
            "b": A.Key(lambda r: int(A.need(other).read_text()), "other.txt", "cmd", "estimate", 0),
            "c": A.Key(lambda r: A.need(tmp_path / "missing").read_text(), "missing", "cmd",
                       "measured")}
    data = {"c": {"value": 3, "status": "measured", "rev": "r0", "date": "d0"}}
    changed, stale = A.collect(data, tmp_path, keys, rev="r1", date="d1")
    assert changed == ["a", "b"] and list(stale) == ["c"]
    assert data["c"]["value"] == 3 and data["c"]["rev"] == "r0"  # kept as is
    src.write_text("2.5")
    changed, _ = A.collect(data, tmp_path, keys, rev="r2", date="d2")
    assert changed == ["a"]
    assert data["a"]["value"] == 2.5 and data["a"]["rev"] == "r2"
    assert data["b"]["rev"] == "r1"  # untouched key keeps its revision


def test_render_file_rev_only_on_change(env, tmp_path, monkeypatch):
    monkeypatch.setattr(A, "git_rev", lambda root=None: "new")
    data_path = tmp_path / "chiffres.json"
    data_path.write_text(json.dumps(DATA))
    kw = {"article": env["article"], "data_path": data_path, "root": env["root"],
          "figures": env["figures"], "suivi_path": env["suivi_path"]}
    assert A.render_file(**kw)
    assert json.loads(data_path.read_text())["_rev"]["rev"] == "new"
    monkeypatch.setattr(A, "git_rev", lambda root=None: "newer")
    assert not A.render_file(**kw)  # nothing changed: same revision stamp
    assert json.loads(data_path.read_text())["_rev"]["rev"] == "new"


def test_repository_article_up_to_date():
    """The versioned article passes --check (same as make ci)."""
    assert A.check(A.ARTICLE.read_text(), A.load()) == []


def test_repository_sources():
    """Every key of CLES reads its source in the repository (results/ is versioned)."""
    data = {}
    _, stale = A.collect(data, ROOT, rev="r", date="d")
    assert stale == {}
    assert data["map.voc.int8"]["value"] == pytest.approx(55.66)
    assert data["map.voc.identical"]["status"] == "csim"
    assert data["perf.kv260.v2.ms"]["status"] == "projection"
