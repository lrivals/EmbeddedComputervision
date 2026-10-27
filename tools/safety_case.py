"""Checks of the assurance case in docs/safety/ (M31, T31.4).

    python tools/safety_case.py            # lists the evidence leaves and their paths
    python tools/safety_case.py --check    # CI: fails on a dangling leaf or a missing file

What is checked:
- every evidence leaf `Sn` of the mermaid tree of `argument.md` has a row in its evidence
  table, and every row is a leaf of the tree;
- the path of each row (relative to the root of the repo) exists;
- every relative markdown link of the files of `docs/safety/` points to an existing file.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAFETY = ROOT / "docs" / "safety"
ARGUMENT = SAFETY / "argument.md"

_MERMAID = re.compile(r"```mermaid\n(.*?)```", re.S)
_LEAF = re.compile(r"\b(S\d+)\[")
_ROW = re.compile(r"^\| (S\d+) \|[^|]*\| `([^`]+)` \|", re.M)
_LINK = re.compile(r"\]\(([^)\s]+)\)")


def leaves(text):
    """Evidence nodes `Sn[...]` declared in the mermaid blocks."""
    return {m for block in _MERMAID.findall(text) for m in _LEAF.findall(block)}


def rows(text):
    """{Sn: path} of the evidence table."""
    return dict(_ROW.findall(text))


def links(path):
    """Relative link targets of a markdown file (anchors and URLs left out)."""
    out = []
    for t in _LINK.findall(path.read_text()):
        if re.match(r"[a-z]+:", t) or t.startswith("#"):
            continue
        out.append(t.split("#", 1)[0])
    return out


def check(root=ROOT):
    """List of problems (empty when the case is consistent)."""
    safety = root / "docs" / "safety"
    argument = safety / "argument.md"
    problems = []
    if not argument.exists():
        return [f"{argument.relative_to(root)} missing"]
    text = argument.read_text()
    tree, table = leaves(text), rows(text)
    if not tree:
        problems.append("argument.md: no evidence leaf in the mermaid tree")
    for s in sorted(tree - table.keys(), key=lambda s: int(s[1:])):
        problems.append(f"argument.md: leaf {s} has no row in the evidence table")
    for s in sorted(table.keys() - tree, key=lambda s: int(s[1:])):
        problems.append(f"argument.md: row {s} is not a leaf of the tree")
    for s, p in table.items():
        if not (root / p).exists():
            problems.append(f"argument.md: {s} → {p} does not exist")
    for md in sorted(safety.glob("*.md")):
        for t in links(md):
            if t and not (md.parent / t).exists():
                problems.append(f"{md.relative_to(root)}: link {t} does not exist")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    problems = check()
    if not args.check:
        for s, p in sorted(rows(ARGUMENT.read_text()).items(), key=lambda kv: int(kv[0][1:])):
            print(f"{s:4} {p}")
    for p in problems:
        print(f"MISSING {p}", file=sys.stderr)
    if args.check and problems:
        sys.exit(1)
    if args.check:
        print(f"safety case: {len(rows(ARGUMENT.read_text()))} evidence leaves, all found")


if __name__ == "__main__":
    main()
