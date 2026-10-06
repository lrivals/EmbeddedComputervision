"""Figures du développement (M13, section E) : chronologie, avancement, code, tests."""

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from tools.figures import ROOT, MissingSource, figure
from tools.figures import style as st

TASKS = ROOT / "docs" / "tasks"
MILESTONE = re.compile(r"^(M\d+(?:\.\d+)?) : ")


def _git(*args):
    if not (ROOT / ".git").exists() or not shutil.which("git"):
        raise MissingSource("dépôt git absent")
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                          text=True).stdout


def _mkey(m):
    return tuple(int(x) for x in m[1:].split("."))


def milestone_files(tasks=None):
    """{jalon: fichier docs/tasks/M*.md}."""
    out = {}
    for p in Path(tasks or TASKS).glob("M*.md"):
        m = re.match(r"(M\d+(?:\.\d+)?)-", p.name)
        if m:
            out[m.group(1)] = p
    return dict(sorted(out.items(), key=lambda kv: _mkey(kv[0])))


# --- T13.27 : chronologie -----------------------------------------------------------------

def load_commits():
    """[(date, sujet)] du plus ancien au plus récent (`git log --date=iso`)."""
    rows = []
    for line in _git("log", "--reverse", "--date=iso-strict", "--format=%ad|%s").splitlines():
        date, _, subject = line.partition("|")
        rows.append((datetime.fromisoformat(date), subject))
    if not rows:
        raise MissingSource("aucun commit")
    return rows


def milestone_spans(commits):
    """{jalon: (début, fin)} : un commit « Mx : » ferme le jalon ouvert au commit précédent."""
    spans = {}
    for k, (date, subject) in enumerate(commits):
        m = MILESTONE.match(subject)
        if m:
            start = commits[k - 1][0] if k else date
            spans.setdefault(m.group(1), (start, date))
    return spans


def plot_chronology(commits, spans, pending, out_dir, name="chronologie"):
    plt = st.plt()
    import matplotlib.dates as mdates

    names = sorted(spans, key=_mkey) + pending
    fig, ax = plt.subplots(figsize=(st.FULL, 0.32 * len(names) + 1.6))
    end = commits[-1][0]
    for y, m in enumerate(names):
        if m in spans:
            a, b = spans[m]
            width = max((b - a).total_seconds() / 86400, 1 / 48)
            ax.barh(y, width, left=mdates.date2num(a), height=0.6, color=st.PALETTE[0])
            ax.text(mdates.date2num(b) + 0.01, y, b.strftime(" %d/%m %Hh%M"), va="center",
                    fontsize=7, color=st.INK2)
        else:
            ax.barh(y, 0.08, left=mdates.date2num(end) + 0.02, height=0.6, color=st.GREY,
                    hatch="///", edgecolor="white")
            ax.text(mdates.date2num(end) + 0.11, y, " non commencé", va="center", fontsize=7,
                    color=st.MUTED)
    for date, subject in commits:
        if subject.startswith("docs"):
            ax.axvline(mdates.date2num(date), color=st.PALETTE[1], lw=0.8, ls=":")
    ax.plot([], [], color=st.PALETTE[1], lw=0.8, ls=":", label="commit de documentation")
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Chronologie des jalons (du commit précédent au commit « Mx : »)")
    ax.legend(loc="lower left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("chronologie", "projet", "T13.27",
        "Gantt des jalons reconstruit depuis les commits « Mx : », avec les commits de "
        "documentation en pointillé et les jalons non commencés en gris.",
        "git log --date=iso")
def chronologie(out_dir):
    commits = load_commits()
    spans = milestone_spans(commits)
    pending = [m for m in milestone_files() if m not in spans and m.split(".")[0] not in spans]
    return plot_chronology(commits, spans, pending, out_dir)


# --- T13.28 : avancement ------------------------------------------------------------------

STATUS = {"x": "faite", "~": "partielle", " ": "ouverte"}
CATEGORIES = ("faite", "partielle", "vérifiée sur PC, carte en attente", "ouverte")
PC_WORDS = ("C-sim", "sim", "PC", "outillés", "estimations", "en attente")


def load_tasks(tasks=None):
    """{jalon: [(id tâche, statut)]} des titres `### [x]`, `### [~]`, `### [ ]`."""
    out = {}
    for m, p in milestone_files(tasks).items():
        out[m] = re.findall(r"^### \[(.)\] (T[\d.]+)", p.read_text(), flags=re.M)
        out[m] = [(t, STATUS.get(s, "ouverte")) for s, t in out[m]]
    return out


def _expand(ids):
    """« T6.0-T6.3 » → T6.0 … T6.3 ; identifiants isolés gardés."""
    out = set()
    for a, b in re.findall(r"(T[\d.]+\d)(?:-(T[\d.]+\d))?", ids):
        out.add(a)
        if b:
            pa, pb = a.rsplit(".", 1), b.rsplit(".", 1)
            if pa[0] == pb[0]:
                out.update(f"{pa[0]}.{i}" for i in range(int(pa[1]), int(pb[1]) + 1))
    return out


def load_tracking(readme=None):
    """Tableau `## Suivi` : {jalon: (tâches, faites, note)}."""
    readme = Path(readme or TASKS / "README.md")
    if not readme.exists():
        raise MissingSource(f"{readme} absent")
    text = readme.read_text().split("## Suivi", 1)
    if len(text) < 2:
        raise MissingSource("README sans section Suivi")
    out = {}
    for m in re.finditer(r"^\| (M[\d.]+) \| (\d+)[^|]* \| (\d+)([^|]*)\|", text[1], flags=re.M):
        out[m.group(1)] = (int(m.group(2)), int(m.group(3)), m.group(4).strip(" ()"))
    return out


def progress(tasks, tracking):
    """{jalon: {catégorie: nombre}} ; « vérifiée sur PC » : tâche ouverte citée par la note
    du suivi avec un mot-clé (C-sim, sim, PC…)."""
    out = {}
    for m, rows in tasks.items():
        note = tracking.get(m, (0, 0, ""))[2]
        pc = _expand(note) if any(w in note for w in PC_WORDS) else set()
        counts = dict.fromkeys(CATEGORIES, 0)
        for t, s in rows:
            if s == "ouverte" and t in pc:
                s = CATEGORIES[2]
            counts[s] += 1
        out[m] = counts
    return out


def tracking_mismatches(tasks, tracking):
    """[(jalon, (tâches, faites) parsées, (tâches, faites) du README)] en désaccord."""
    bad = []
    for m, rows in tasks.items():
        got = (len(rows), sum(s == "faite" for _, s in rows))
        if m in tracking and got != tracking[m][:2]:
            bad.append((m, got, tracking[m][:2]))
    return bad


def plot_progress(prog, mismatches, out_dir, name="avancement"):
    plt = st.plt()
    names = list(prog)
    fig, ax = plt.subplots(figsize=(st.FULL, 0.3 * len(names) + 1.4))
    left = np.zeros(len(names))
    colors = (st.PALETTE[5], st.PALETTE[3], st.PALETTE[2], st.GREY)
    for k, cat in enumerate(CATEGORIES):
        v = np.array([prog[m][cat] for m in names])
        ax.barh(range(len(names)), v, left=left, height=0.62, color=colors[k],
                hatch=("", "..", "//", "")[k], edgecolor="white", linewidth=0.6, label=cat)
        left += v
    for y, m in enumerate(names):
        done = prog[m]["faite"]
        ax.text(left[y] + 0.4, y, f"{done}/{int(left[y])}", va="center", fontsize=7,
                color=st.INK2)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("tâches")
    ax.set_title("Avancement par jalon (titres ### [x] / [~] / [ ] de docs/tasks/)")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y", visible=False)
    if mismatches:
        st.note(ax, "écarts avec le suivi du README : " + ", ".join(
            f"{m} {g[1]}/{g[0]} ≠ {r[1]}/{r[0]}" for m, g, r in mismatches), "lower right")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("avancement", "projet", "T13.28",
        "Tâches faites, partielles, vérifiées sur PC (carte en attente) et ouvertes, par jalon.",
        "docs/tasks/M*.md, tableau de suivi de docs/tasks/README.md")
def avancement(out_dir):
    tasks = load_tasks()
    if not tasks:
        raise MissingSource("aucun docs/tasks/M*.md")
    tracking = load_tracking()
    return plot_progress(progress(tasks, tracking), tracking_mismatches(tasks, tracking),
                         out_dir)


# --- T13.29 : volume de code --------------------------------------------------------------

FOLDERS = ("python", "tools", "cpp", "hls", "sw", "docs", "results")


def load_code_history():
    """(dates, sujets, {dossier: lignes nettes cumulées par commit}) de `git log --numstat`."""
    out = _git("log", "--reverse", "--numstat", "--date=iso-strict", "--format=@%ad|%s")
    dates, subjects, rows = [], [], []
    cur = dict.fromkeys(FOLDERS + ("autres",), 0)
    for line in out.splitlines():
        if line.startswith("@"):
            if dates:
                rows.append(dict(cur))
            d, _, s = line[1:].partition("|")
            dates.append(datetime.fromisoformat(d))
            subjects.append(s)
        elif line.strip():
            add, rem, path = line.split("\t", 2)
            if add == "-":
                continue  # binaire
            if "=>" in path:  # renommage : le chemin d'arrivée compte
                path = re.sub(r"\{[^}]*=> ([^}]*)\}", r"\1", path).split(" => ")[-1]
            top = path.split("/", 1)[0]
            cur[top if top in FOLDERS else "autres"] += int(add) - int(rem)
    if dates:
        rows.append(dict(cur))
    if not rows:
        raise MissingSource("aucun commit")
    return dates, subjects, {k: np.array([r[k] for r in rows]) for k in rows[0]}


def plot_code(subjects, series, out_dir, name="code"):
    plt = st.plt()
    keys = [k for k in series if series[k][-1] > 0]
    x = np.arange(len(subjects))
    fig, ax = plt.subplots(figsize=(st.FULL, 4.4))
    ax.stackplot(x, [series[k] / 1e3 for k in keys], colors=st.PALETTE[:len(keys)],
                 labels=[f"{k}/ : {series[k][-1]:,} lignes".replace(",", " ") for k in keys],
                 edgecolor="white", linewidth=0.6)
    labels = [MILESTONE.match(s).group(1) if MILESTONE.match(s) else
              ("docs" if s.startswith("docs") else "") for s in subjects]
    ax.set_xticks(x, labels, rotation=60, fontsize=7)
    ax.set_xlim(0, len(x) - 1)
    ax.set_ylabel("milliers de lignes")
    total = sum(series[k][-1] for k in keys)
    n = f"{total:,}".replace(",", " ")
    ax.set_title(f"Volume de code par dossier, commit par commit ({n} lignes)")
    ax.legend(loc="upper left", ncol=2, reverse=True)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("code", "projet", "T13.29",
        "Lignes par dossier (python, tools, cpp, hls, sw, docs, results) commit par commit.",
        "git log --numstat")
def code(out_dir):
    _, subjects, series = load_code_history()
    return plot_code(subjects, series, out_dir)


# --- T13.30 : tests -----------------------------------------------------------------------

def load_pytest_counts():
    """{module de test: nombre de tests} de `pytest --collect-only -q`, slow compris."""
    import sys

    if not (ROOT / "python" / "tests").exists():
        raise MissingSource("python/tests absent")
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", ""],
                       cwd=ROOT / "python", capture_output=True, text=True)
    counts = {}
    for line in r.stdout.splitlines():
        m = re.match(r"tests/(test_\w+)\.py::", line)
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    if not counts:
        raise MissingSource("pytest --collect-only n'a rien collecté")
    return counts


def load_ctest_counts():
    """{répertoire de build: nombre de tests ctest} des builds C++ présents."""
    if not shutil.which("ctest"):
        return {}
    out = {}
    for d in ("build/golden", "build/hls", "build/sw"):
        if (ROOT / d / "CTestTestfile.cmake").exists():
            r = subprocess.run(["ctest", "-N"], cwd=ROOT / d, capture_output=True, text=True)
            m = re.search(r"Total Tests: (\d+)", r.stdout)
            if m:
                out[d] = int(m.group(1))
    return out


PYTEST_XML = ROOT / "build" / "figures" / "pytest.xml"


def _junit_cases(path):
    import xml.etree.ElementTree as ET

    for case in ET.parse(path).getroot().iter("testcase"):
        yield case.get("classname") or "", case.get("name") or "", float(case.get("time") or 0)


def load_test_durations():
    """{module ou « ctest <build> »: secondes} des rapports JUnit de `make test-durations`
    (pytest --junitxml, ctest --output-junit) ; vide si la suite n'a pas été relancée."""
    out = {}
    if PYTEST_XML.exists():
        for cls, _, t in _junit_cases(PYTEST_XML):
            m = re.search(r"(test_\w+)", cls)
            if m:
                out[m.group(1)] = out.get(m.group(1), 0.0) + t
    for d in ("build/golden", "build/hls", "build/sw"):
        x = ROOT / d / "ctest.xml"
        if x.exists():
            out[f"ctest {d}"] = sum(t for _, _, t in _junit_cases(x))
    return out


def plot_tests(py, cpp, out_dir, durations=None, name="tests"):
    plt = st.plt()
    items = sorted(py.items(), key=lambda kv: kv[1]) + [(f"ctest {k}", v) for k, v in cpp.items()]
    durations = durations or {}
    ncol = 2 if durations else 1
    fig, axes = plt.subplots(1, ncol, figsize=(st.FULL, 0.2 * len(items) + 1.2), sharey=True,
                             squeeze=False)
    ax = axes[0][0]
    y = np.arange(len(items))
    colors = [st.PALETTE[1] if k.startswith("ctest") else st.PALETTE[0] for k, _ in items]
    ax.barh(y, [v for _, v in items], 0.7, color=colors)
    ax.set_yticks(y, [k for k, _ in items], fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("tests (log)")
    ax.set_title(f"Tests : {sum(py.values())} pytest (slow compris, {len(py)} modules) + "
                 f"{sum(cpp.values())} ctest", fontsize=9)
    ax.grid(axis="y", visible=False)
    if durations:
        ax2 = axes[0][1]
        secs = [durations.get(k) for k, _ in items]
        ax2.barh(y, [max(t or 0, 1e-3) for t in secs], 0.7, color=colors)
        for yi, t in zip(y, secs):
            if t is None:
                ax2.text(1.2e-3, yi, "non relancé", va="center", fontsize=6, color=st.MUTED)
        ax2.set_xscale("log")
        ax2.set_xlabel("durée (s, log)")
        ax2.set_title(f"Durée : {sum(t for t in secs if t) / 60:.1f} min au total "
                      "(make test-durations)", fontsize=9)
        ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("tests", "projet", "T13.30",
        "Nombre de tests par module Python (pytest, slow compris) et par build C++ (ctest), "
        "et leur durée quand `make test-durations` a été lancé.",
        "pytest --collect-only -q, ctest -N, build/figures/pytest.xml, build/*/ctest.xml")
def tests(out_dir):
    return plot_tests(load_pytest_counts(), load_ctest_counts(), out_dir,
                      durations=load_test_durations())


# --- T13.30 (suite) : graphe des dépendances ----------------------------------------------

NODE = re.compile(r'(\w+)(?:\[\"?([^\]\"]+)\"?\])?')
ARROW = re.compile(r'\s*(-->|-\.->|--\s*"[^"]*"\s*-->)\s*')


def load_mermaid(readme=None):
    """Graphe mermaid de docs/tasks/README.md : ({id: libellé}, [(src, dst, libellé, pointillé)])."""
    readme = Path(readme or TASKS / "README.md")
    if not readme.exists():
        raise MissingSource(f"{readme} absent")
    m = re.search(r"```mermaid\n(.*?)```", readme.read_text(), flags=re.S)
    if not m:
        raise MissingSource("README sans graphe mermaid")
    nodes, edges = {}, []
    for line in m.group(1).splitlines()[1:]:
        parts = ARROW.split(line.strip())
        if len(parts) < 3:
            continue
        groups = []
        for tok in parts[::2]:
            ids = []
            for item in tok.split("&"):
                n = NODE.fullmatch(item.strip())
                nodes.setdefault(n.group(1), n.group(1))
                if n.group(2):
                    nodes[n.group(1)] = n.group(2)
                ids.append(n.group(1))
            groups.append(ids)
        for arrow, a, b in zip(parts[1::2], groups, groups[1:]):
            lab = re.search(r'"([^"]*)"', arrow)
            for s in a:
                for d in b:
                    edges.append((s, d, lab.group(1) if lab else "", "-.->" in arrow))
    return nodes, edges


def dependency_dot(nodes, edges, prog):
    def color(label):
        c = prog.get(label)
        if not c:
            return "white", st.MUTED
        total = sum(c.values()) or 1
        done = c["faite"] / total
        fill = (st.PALETTE[5] if done == 1 else st.PALETTE[3] if done > 0 or c["partielle"]
                or c[CATEGORIES[2]] else st.GREY)
        return fill, f"{c['faite']}/{total}"

    lines = ["digraph jalons {", "  rankdir=LR; nodesep=0.3; ranksep=0.45;",
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
             '  edge [color="#8a8984", fontname="Helvetica", fontsize=8, arrowsize=0.7];']
    for nid, label in nodes.items():
        fill, sub = color(label)
        text = f"{label}\\n{sub}" if sub != st.MUTED else label
        fc = "white" if fill == st.PALETTE[5] else st.INK
        lines.append(f'  {nid} [label="{text}", fillcolor="{fill}", fontcolor="{fc}"];')
    for s, d, lab, dashed in edges:
        attrs = [f'label="{lab}"'] if lab else []
        if dashed:
            attrs.append("style=dashed")
        lines.append(f"  {s} -> {d}{' [' + ', '.join(attrs) + ']' if attrs else ''};")
    lines.append('  label="vert : jalon fait · ambre : commencé ou vérifié sur PC · gris : à faire '
                 '· n/N : tâches faites"; labelloc=b; fontname="Helvetica"; fontsize=10;')
    lines.append("}")
    return "\n".join(lines) + "\n"


@figure("dependances", "projet", "T13.30",
        "Graphe des jalons du README (défini une seule fois, en mermaid), coloré selon "
        "l'avancement de T13.28.",
        "docs/tasks/README.md (mermaid et suivi), docs/tasks/M*.md (Graphviz dot)")
def dependances(out_dir):
    dot = shutil.which("dot")
    if not dot:
        raise MissingSource("Graphviz (dot) absent")
    nodes, edges = load_mermaid()
    tasks = load_tasks()
    src = dependency_dot(nodes, edges, progress(tasks, load_tracking()))
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "dependances.png"
    subprocess.run([dot, "-Tpng", "-Gdpi=150", "-o", str(png)], input=src, text=True, check=True)
    return [png]
