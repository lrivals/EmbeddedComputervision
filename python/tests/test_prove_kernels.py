"""T28.2: SMT proof of the output-stage kernels (tools/prove_kernels.py), skipped without z3.

The full proof (n = 1…31, every export) takes minutes: here a few shifts, and the two
mutants (no rounding term; 13·y on 32 bits, the T28.1 finding) must give a counter-example."""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("z3")

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "prove_kernels.py"


def _run(*args):
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True,
                          timeout=900)


def test_proof_holds_on_some_shifts():
    r = _run("--shifts", "30-31")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "8/8 obligations proved" in r.stdout


@pytest.mark.parametrize("mutant, shift", [("rounding", "12"), ("leaky32", "1")])
def test_mutant_gives_counter_example(mutant, shift):
    r = _run("--mutant", mutant, "--shifts", shift)
    assert r.returncode == 0, r.stdout + r.stderr  # exit 0: the mutant was caught
    assert "COUNTER-EXAMPLE" in r.stdout


def test_cpp_model_matches_spec_on_points():
    """The hand-written z3 model evaluates like the Python operators on concrete values."""
    import numpy as np
    import z3

    sys.path.insert(0, str(ROOT / "tools"))
    import prove_kernels as pk

    from yolo.quant import int_layers as il

    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(1, 32))
        m0 = int(rng.integers(1, 1 << n))
        acc = int(rng.integers(-(1 << 31), 1 << 31))
        y = z3.simplify(pk.cpp_requant(z3.BitVecVal(acc, 32), z3.BitVecVal(m0, 32), n))
        want = int(il.rshift_round(np.int64(acc) * m0, n))
        assert y.as_signed_long() == want
        v = z3.simplify(pk.cpp_leaky(z3.BitVecVal(want, 32))).as_signed_long()
        assert v == int(il.leaky_int(want))
