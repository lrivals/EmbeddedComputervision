"""M31: dangerous failure rate budget (T31.3)."""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import failure_budget as B  # noqa: E402

DEEL = """# x

<!-- m27:deel-prewid -->
## From per-image errors to failures per hour (T27.5)

| Model | Dangerous failures / h | vs SIL 1 | vs SIL 2 | vs SIL 3 |
|---|---|---|---|---|
| independent | 0.0206 | +3.3 | +4.3 | +5.3 |
| measured | 25.2 | +6.4 | +7.4 | +8.4 |
<!-- /m27:deel-prewid -->
"""


def _nominal(**kw):
    v = {i.name: i.value for i in B.inputs({"measured": 25.2, "independent": 0.0206})}
    return {**v, **kw}


def test_deel_rates(tmp_path):
    p = tmp_path / "ps.md"
    p.write_text(DEEL)
    assert B.deel_rates(p) == {"independent": 0.0206, "measured": 25.2}


def test_deel_rates_of_the_repo():
    r = B.deel_rates()
    assert r["measured"] > r["independent"] > 0


def test_budget_values():
    b = B.budget(_nominal())
    assert b["perception"] == pytest.approx(25.2 * (1 - 0.105))
    assert b["execution"] == pytest.approx(1e-5 * 0.1 * 0.24)
    assert b["deadline"] == pytest.approx(1e-5 * 0.1)
    assert b["total"] == pytest.approx(b["perception"] + b["execution"] + b["deadline"])
    # full coverage, perfect watchdog and no silent fault: nothing left
    zero = B.budget(_nominal(coverage=1.0, watchdog=1.0, silent=0.0))
    assert zero["total"] == 0


def test_gaps():
    g = B.gaps(1e-4)
    assert g["SIL 1"] == pytest.approx(1) and g["SIL 3"] == pytest.approx(3)
    assert g["PL e"] == pytest.approx(3) and g["PL d"] == pytest.approx(2)
    assert g["PL c"] == pytest.approx(math.log10(1e-4 / 3e-6))
    assert B.gaps(0)["SIL 2"] == -math.inf


def test_sensitivity_monotone():
    ins = B.inputs({"measured": 25.2, "independent": 0.0206})
    base, rows = B.sensitivity(ins)
    assert base == pytest.approx(B.budget(_nominal())["total"])
    by = {i.name: (lo, hi) for i, lo, hi, _, _ in rows}
    assert by["coverage"][0] > base > by["coverage"][1]  # more coverage, fewer failures
    assert by["vote"][0] < base == pytest.approx(by["vote"][1])
    assert by["model"][0] < base < by["model"][1]
    assert by["presence"][1] == pytest.approx(base * 10)  # presence 0.1 → 1 scales all


def test_write(tmp_path, monkeypatch):
    ps = tmp_path / "ps.md"
    ps.write_text(DEEL)
    out = tmp_path / "budget.md"
    monkeypatch.setattr(B, "RESULTS", out)
    B.main(["--person-safety", str(ps), "--write"])
    text = out.read_text()
    assert text.startswith("# Dangerous failure rate budget")
    assert "<!-- m31:budget -->" in text and "| SIL 2 | 1e-06 | +7.4 |" in text
    assert "**Provisional**" in text
    B.main(["--person-safety", str(ps), "--write"])  # replaced, not appended
    assert out.read_text().count("<!-- m31:budget -->") == 1


def test_set_rejects_unknown_and_write():
    with pytest.raises(SystemExit):
        B.main(["--set", "nope=1"])
    with pytest.raises(SystemExit):
        B.main(["--set", "vote=0.1", "--write"])


def test_committed_budget_up_to_date():
    text = B.render(B.inputs(B.deel_rates()))
    assert text.rstrip() in B.RESULTS.read_text(), "python tools/failure_budget.py --write"
