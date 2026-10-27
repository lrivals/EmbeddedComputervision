"""Dangerous failure rate budget of the zone monitor (M31, T31.3).

    python tools/failure_budget.py                    # prints the budget
    python tools/failure_budget.py --write            # section of results/failure_budget.md
    python tools/failure_budget.py --set coverage=0.43 --set vote=0.1   # what-if

The dreaded event E1 (docs/safety/requirements.md): a person in the zone, not reported during
N frames. Its rate per hour is the sum of three families (T25.10), assumed independent:

    perception  = model · (1 − coverage) · vote
    execution   = hw_rate · presence · silent · hw_miss
    deadline    = hang_rate · presence · (1 − watchdog)

`model` is the detector-alone rate of T27.5, read from the `deel` section of
results/person_safety.md, so a new run of T27.5 moves the budget. The other inputs are the
measured shares of M25, M29 and M39 where they exist, or written assumptions; each carries a
status (measured / provisional / assumption / missing) and a range for the sensitivity table.
"""

import argparse
import math
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from person_safety import RESULTS as PERSON_SAFETY  # noqa: E402
from person_safety import SIL_PFH, sil_gap, table  # noqa: E402

RESULTS = ROOT / "results" / "failure_budget.md"
# ISO 13849-1 table 2 (to read): upper bound of the average probability of a dangerous
# failure per hour of each PL.
PL_PFH = {"c": 3e-6, "d": 1e-6, "e": 1e-7}
TARGET = ("SIL 2", SIL_PFH[2])  # to be confirmed by ARCYS (requirements.md, SR-04)


@dataclass(frozen=True)
class Input:
    name: str
    value: float
    low: float
    high: float
    what: str
    source: str
    status: str  # measured / provisional / assumption / missing


def deel_rates(path=None):
    """{"independent": /h, "measured": /h} of the T27.5 table of results/person_safety.md."""
    text = Path(path or PERSON_SAFETY).read_text()
    a, b = text.index("<!-- m27:deel-"), text.index("<!-- /m27:deel-")
    rows = re.findall(r"^\| (independent|measured) \| ([0-9.e+-]+) \|", text[a:b], re.M)
    if not rows:
        raise ValueError(f"no T27.5 rates in {path or PERSON_SAFETY}")
    return {k: float(v) for k, v in rows}


def inputs(rates):
    m, ind = rates["measured"], rates.get("independent", rates["measured"] * 1e-3)
    return [
        Input("model", m, ind, m * 10,
              "E1 runs per hour, detector alone (presence 10 %, N = 5)",
              "T27.5, measured burst rate (`results/person_safety.md`); low = independent misses",
              "measured"),
        Input("coverage", 0.105, 0.0, 0.9,
              "share of the E1 runs flagged by the monitor (→ `UNSAFE`)",
              "T29.1 / T39.1 reduced run: 10.5 % of missed persons flagged at 5 % alarms "
              "(`results/failure_prediction.md`), taken per run (flags of a run fully "
              "correlated); 0.43 if independent over 5 frames; 0.9 = SR-05", "provisional"),
        Input("vote", 1.0, 0.01, 1.0,
              "factor of the temporal vote on the remaining runs",
              "T29.3 not done: no gain counted; 0.01 = optimistic", "missing"),
        Input("hw_rate", 1e-5, 1e-7, 1e-4,
              "upsets per hour in the accelerator data path (≈ 10⁴ FIT)",
              "assumption; to replace by the device reliability data (to read)", "assumption"),
        Input("silent", 0.24, 0.0, 0.81,
              "share of the upsets that change the output without ABFT alarm",
              "T25.7 C-sim, mean over in / w / out / q of v3 (13, 0, 0, 81 %); high = `q` alone "
              "(`results/faults.md`); T25.4 not done", "provisional"),
        Input("hw_miss", 1.0, 0.01, 1.0,
              "share of the silent corruptions that hide a present person for N frames",
              "T25.1 / T25.6 not run: worst case 1; a buffer flip lasts one tile, a DDR or "
              "configuration fault until reload", "missing"),
        Input("hang_rate", 1e-5, 1e-7, 1e-3,
              "frames lost or late per hour (hang, DDR stall, ARM pre-emption)",
              "assumption; T30.2 / T30.3 not measured", "assumption"),
        Input("watchdog", 0.0, 0.0, 1.0,
              "share of the late or missing frames turned into `UNSAFE`",
              "T29.5 not written: 0; the bound exists for Tiny-YOLOv2 (116 ms, T30.0)", "missing"),
        Input("presence", 0.1, 0.01, 1.0,
              "share of the time a person is in the zone",
              "assumption of T27.5 (already inside `model`)", "assumption"),
    ]


def budget(v):
    """Contributions per hour from a {name: value} dict."""
    p = v["model"] * (1 - v["coverage"]) * v["vote"]
    x = v["hw_rate"] * v["presence"] * v["silent"] * v["hw_miss"]
    d = v["hang_rate"] * v["presence"] * (1 - v["watchdog"])
    return {"perception": p, "execution": x, "deadline": d, "total": p + x + d}


def gaps(rate):
    """Orders of magnitude above each SIL and PL bound (positive: the bound is missed)."""
    out = {f"SIL {k}": g for k, g in sil_gap(rate).items()}
    out.update({f"PL {k}": (math.log10(rate / b) if rate > 0 else -math.inf)
                for k, b in PL_PFH.items()})
    return out


def sensitivity(ins):
    """Per input: total at its low and high value, the others at their nominal value."""
    nominal = {i.name: i.value for i in ins}
    base = budget(nominal)["total"]
    rows = []
    for i in ins:
        if i.name == "presence":  # already folded into `model` by T27.5
            lo_t = budget({**nominal, "presence": i.low,
                           "model": nominal["model"] * i.low / i.value})["total"]
            hi_t = budget({**nominal, "presence": i.high,
                           "model": nominal["model"] * i.high / i.value})["total"]
        else:
            lo_t = budget({**nominal, i.name: i.low})["total"]
            hi_t = budget({**nominal, i.name: i.high})["total"]
        rows.append((i, lo_t, hi_t, math.log10(lo_t / base), math.log10(hi_t / base)))
    return base, rows


def fmt(x):
    return f"{x:.3g}"


def render(ins):
    v = {i.name: i.value for i in ins}
    b = budget(v)
    g = gaps(b["total"])
    missing = [i.name for i in ins if i.status in ("missing", "provisional")]
    name, target = TARGET
    need = b["total"] / target
    a = [[f"`{i.name}`", fmt(i.value), f"{fmt(i.low)} – {fmt(i.high)}", i.what, i.source,
          i.status] for i in ins]
    contrib = [[k, fmt(b[k]), f"{100 * b[k] / b['total']:.3g} %" if b["total"] else "–"]
               for k in ("perception", "execution", "deadline")]
    contrib.append(["**total**", f"**{fmt(b['total'])}**", "100 %"])
    gap_rows = [[k, fmt(SIL_PFH[int(k[-1])] if k.startswith("SIL") else PL_PFH[k[-1]]),
                 f"{g[k]:+.1f}"] for k in g]
    base, sens = sensitivity(ins)
    s_rows = [[f"`{i.name}`", f"{fmt(i.low)} → {fmt(lo)}", f"{fmt(i.high)} → {fmt(hi)}",
               f"{dl:+.1f} / {dh:+.1f}"] for i, lo, hi, dl, dh in sens]
    body = [
        "**Provisional**: the inputs " + ", ".join(f"`{n}`" for n in missing) + " are not "
        "measured yet or come from reduced runs; the budget is regenerated when T25.4, "
        "T29.1, T29.3, T29.5 and T27.7 (monitor rows) land." if missing else "",
        "",
        "Dangerous failure (E1, [requirements.md](../docs/safety/requirements.md)): a person "
        "in the zone not reported during N = 5 frames. Rates per hour of operation.",
        "",
        "### Inputs",
        "",
        table(["Input", "Value", "Range", "What", "Source", "Status"], a),
        "",
        "### Independence assumptions",
        "",
        "- The three families add up (rare events, no common cause): a fault does not make "
        "the model miss more often than its own rate, and a hang does not coincide with a "
        "fault.",
        "- The monitor coverage and the vote multiply: the vote acts on the runs the monitor "
        "did not flag as if they were a random sample of the runs.",
        "- `model` is the **measured** burst rate (runs of 5 missed frames counted on video), "
        "not the product of per-frame misses: frame-to-frame correlation is inside it (×1,240 "
        "against independence, T27.4).",
        "- A detected fault (ABFT), a flag of the monitor and a late frame are safe only "
        "once the fallback (T29.5) answers them with `UNSAFE`: the budget counts them as "
        "safe already, except the watchdog (`watchdog` = 0 until T29.5).",
        "",
        "### Contributions",
        "",
        table(["Family", "Per hour", "Share"], contrib),
        "",
        "### Against the bounds",
        "",
        table(["Bound", "PFH upper bound (/h)", "Gap (orders of magnitude)"], gap_rows),
        "",
        f"Against the {name} target, the architecture still has to gain a factor "
        f"**{need:.2g}** (10^{math.log10(need):.1f}) over the current budget. SIL bounds: "
        "IEC 61508-1 table 3; PL bounds: ISO 13849-1 table 2 (to read).",
        "",
        "### Sensitivity",
        "",
        f"Total per hour with one input at the end of its range, the others nominal "
        f"(nominal {fmt(base)} /h); last column: change in orders of magnitude.",
        "",
        table(["Input", "Low → total", "High → total", "Δ log10 (low / high)"], s_rows),
        "",
        "Reading: the perception term dominates by orders of magnitude, so the hardware and "
        "deadline assumptions change nothing until the misses are cut. The two levers are the "
        "monitor coverage (even 0.9 gives one order) and the temporal vote; neither reaches "
        f"the {name} target alone.",
    ]
    return "\n".join(body).lstrip()


HEADER = """# Dangerous failure rate budget (M31)

Generated by `python tools/failure_budget.py --write` from the T27.5 rate of
[person_safety.md](person_safety.md) and the inputs written in the tool
([docs/tasks/M31-assurance-case.md](../docs/tasks/M31-assurance-case.md), T31.3).
"""


def write_section(body, path=None):
    path = path or RESULTS
    start, end = "<!-- m31:budget -->", "<!-- /m31:budget -->"
    text = path.read_text() if path.exists() else HEADER
    block = f"{start}\n## Budget per hour (T31.3)\n\n{body.rstrip()}\n{end}"
    if start in text:
        a, b = text.index(start), text.index(end) + len(end)
        text = text[:a] + block + text[b:]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"→ {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override an input (what-if; not for --write)")
    ap.add_argument("--person-safety", type=Path, default=None)
    ap.add_argument("--write", action="store_true", help="section of results/failure_budget.md")
    args = ap.parse_args(argv)
    if args.write and args.set:
        ap.error("--write takes the nominal inputs only")
    ins = inputs(deel_rates(args.person_safety))
    by = {i.name: i for i in ins}
    for s in args.set:
        k, _, val = s.partition("=")
        if k not in by:
            ap.error(f"unknown input {k!r} (one of {', '.join(by)})")
        by[k] = replace(by[k], value=float(val), status="what-if")
    ins = list(by.values())
    text = render(ins)
    print(text)
    if args.write:
        write_section(text)


if __name__ == "__main__":
    main()
