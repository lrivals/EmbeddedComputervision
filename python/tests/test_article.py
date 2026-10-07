"""M17: LaTeX article (tools/article.py) — macros of article.tex, generated.tex, --check,
collect from the sources, compilation when xelatex is installed."""

import json
import shutil
import subprocess
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

TEXT = r"""\documentclass{article}
\input{generated}
\begin{document}
\begin{abstract}
The mAP is \chiffre{map.voc.int8}, threshold 0.45. % 1.23 in a comment is fine
\end{abstract}
\section{Introduction}
Narrative with 3.14 outside results. State: \articleetatresume.
\section{Two}
\section{Three}
\section{Inference}
Table: \articletable{t}. Csim: \chiffre{map.voc.csim}.
\articlefig{graphe/graphe_net}
\section{Limits}
\articlestatuslist
\articleetat
Rendered at \articlerev.
\end{document}
"""

DATA = {
    "map.voc.int8": {"value": 55.66, "status": "measured", "nd": 2},
    "map.voc.csim": {"value": 55.66, "status": "csim", "nd": 2},
    "t": {"value": {"header": ["A", "B value"], "rows": [["x_`code`", 1234.5], ["y", -0.5]],
                    "nd": [None, 1], "status": "projection"}, "status": "projection"},
    "_rev": {"rev": "abc1234", "date": "2026-10-07"},
}


@pytest.fixture
def env(tmp_path):
    png = tmp_path / "results" / "figures" / "modeles" / "graphe_net.png"
    png.parent.mkdir(parents=True)
    png.write_bytes(b"png")
    (tmp_path / "suivi.md").write_text(SUIVI)
    article = tmp_path / "docs" / "article" / "article.tex"
    article.parent.mkdir(parents=True)
    article.write_text(TEXT)
    figs = {"graphe": Fig("graphe", "modeles", "Graph of the net → 2ⁿ."),
            "sub": Fig("sub", "resultats", "Subset.", subset=True)}
    return {"root": tmp_path, "article": article, "figures": figs,
            "suivi_path": tmp_path / "suivi.md"}


def _render(text, env, data=DATA):
    return A.render(text, data, env["figures"], env["article"], env["root"], env["suivi_path"])


def _check(text, gen, env, data=DATA):
    return A.check(text, gen, data, env["figures"], env["article"], env["root"],
                   env["suivi_path"])


def test_scan():
    uses = A.scan(TEXT)
    assert (5, "chiffre", "map.voc.int8") in uses
    assert (12, "articletable", "t") in uses
    assert (13, "articlefig", "graphe/graphe_net") in uses
    assert (8, "articleetatresume", "") in uses and (16, "articleetat", "") in uses
    assert A.scan(r"% \chiffre{commented}" + "\n50\\% \\chiffre{a}") == [(2, "chiffre", "a")]


def test_render_generated(env):
    out = _render(TEXT, env)
    assert r"\csname artn:map.voc.int8\endcsname{%" + "\n55.66}" in out
    assert "55.66\\textsuperscript{C-sim}" in out
    assert r"x\_\texttt{code} & 1,234.5 \\" in out and r"y & −0.5 \\" in out
    assert r"\textbf{\begin{tabular}[b]{@{}r@{}}B\\value\end{tabular}}" in out
    assert "Status of every number of this table: projection." in out
    assert "{../../results/figures/modeles/graphe_net.png}" in out
    assert r"\caption{Graph of the net $\to$ 2$^n$. (\texttt{graphe})}" in out
    assert r"\label{fig:graphe/graphe_net}" in out
    assert "% python -m tools.figures graphe" in out
    assert "14 of 15 tasks done across 2 milestones, 1 milestones complete" in out
    assert r"M1 & 9 & 8 & 89\,\% \\" in out
    assert r"rev.~\texttt{abc1234}, 2026-10-07" in out
    assert r"\item \textbf{csim} (1): \texttt{map.voc.csim}" in out
    assert r"\item \textbf{projection} (1): \texttt{t}" in out


def test_status_mark_and_format():
    assert A.render_n({"value": 37.2, "status": "projection", "nd": 1}) == \
        r"37.2\textsuperscript{proj.}"
    assert A.render_n({"value": -0.64, "status": "measured", "nd": 2, "sign": True}) == "−0.64"
    assert A.fmt(0.1, 2, sign=True) == "+0.10"
    assert A.fmt(4952) == "4,952"
    assert A.fmt(-0.001, 2) == "0.00"
    assert A.latex("a_b & 50 % `x_y` #1") == r"a\_b \& 50 \% \texttt{x\_y} \#1"


@pytest.mark.parametrize("tex, why", [
    ("\n\\chiffre{nope}", "line 2: key 'nope' absent"),
    (r"\articletable{map.voc.int8}", "is a number"),
    (r"\chiffre{t}", "is a table"),
    (r"\articlefig{inconnue}", "absent from the registry"),
    (r"\articlefig{sub}", "not published"),
    (r"\articlefig{graphe/absent}", "absent"),
])
def test_render_errors(env, tex, why):
    with pytest.raises(A.ArticleError, match=why):
        _render(tex, env)


def test_check(env):
    gen = _render(TEXT, env)
    assert _check(TEXT, gen, env) == []
    probs = _check(TEXT, gen.replace("{%\n55.66}", "{%\n55.70}", 1), env)
    assert len(probs) == 1 and "generated.tex differs" in probs[0] and "55.70" in probs[0]
    assert _check(TEXT, "", env)  # never rendered
    assert any("absent from chiffres.json" in p for p in _check(TEXT + r"\chiffre{x}", gen, env))


def test_stray_numbers(env):
    assert A.stray_numbers(TEXT) == []  # 3.14 in §1, 0.45 whitelisted, comment ignored
    bad = TEXT.replace("Csim:", r"gain 1.37 \cite[\#013.1]{2024-zhang}, \texttt{p99.99}, "
                       r"$1.5$, §10.4, T17.2. Csim:")
    assert A.stray_numbers(bad) == [(12, "1.37")]
    gen = _render(bad, env)
    assert any("1.37 written by hand" in p for p in _check(bad, gen, env))
    late = TEXT.replace(r"\articlestatuslist", r"\articlestatuslist 2.71")  # section 5: checked
    assert A.stray_numbers(late) == [(15, "2.71")]
    assert A.stray_numbers(late.replace(r"\section{Limits}", r"\appendix")) == []


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
    kw = {"article": env["article"], "generated": tmp_path / "generated.tex",
          "data_path": data_path, "root": env["root"],
          "figures": env["figures"], "suivi_path": env["suivi_path"]}
    assert A.render_file(**kw)
    assert json.loads(data_path.read_text())["_rev"]["rev"] == "new"
    monkeypatch.setattr(A, "git_rev", lambda root=None: "newer")
    assert not A.render_file(**kw)  # nothing changed: same revision stamp
    assert json.loads(data_path.read_text())["_rev"]["rev"] == "new"


def test_repository_article_up_to_date():
    """The versioned article passes --check (same as make ci)."""
    assert A.check(A.ARTICLE.read_text(), A.GENERATED.read_text(), A.load()) == []


def test_repository_sources():
    """Every key of CLES reads its source in the repository (results/ is versioned)."""
    data = {}
    _, stale = A.collect(data, ROOT, rev="r", date="d")
    assert stale == {}
    assert data["map.voc.int8"]["value"] == pytest.approx(55.66)
    assert data["map.voc.identical"]["status"] == "csim"
    assert data["perf.kv260.v2.ms"]["status"] == "projection"


@pytest.mark.skipif(not shutil.which("xelatex"), reason="xelatex absent")
def test_compiles(env, tmp_path):
    """The generated macros compile: numbers, a table, a figure, the status."""
    pytest.importorskip("PIL")
    from PIL import Image

    png = env["root"] / "results" / "figures" / "modeles" / "graphe_net.png"
    Image.new("RGB", (40, 20), "white").save(png)
    tex = TEXT.replace(r"\documentclass{article}", "\\documentclass{article}\n"
                       "\\usepackage{fontspec,graphicx,booktabs,tabularx,array}")
    tex = tex.replace("\\begin{abstract}", "").replace("\\end{abstract}", "")
    env["article"].write_text(tex)
    (env["article"].parent / "generated.tex").write_text(_render(tex, env))
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                        "article.tex"], cwd=env["article"].parent, capture_output=True,
                       text=True, timeout=300, check=False)
    assert r.returncode == 0, r.stdout[-2000:]
    assert (env["article"].parent / "article.pdf").exists()
