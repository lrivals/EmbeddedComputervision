"""T8.1 : statistiques et lecture des rapports de tools/bench_report.py."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
_spec = importlib.util.spec_from_file_location("bench_report", ROOT / "tools" / "bench_report.py")
br = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(br)

# Extraits au format de Vivado 2023.x (report_utilization / report_power, UltraScale+).
UTIL = """
+----------------------------+-------+-------+------------+-----------+-------+
|          Site Type         |  Used | Fixed | Prohibited | Available | Util% |
+----------------------------+-------+-------+------------+-----------+-------+
| CLB LUTs                   | 41234 |     0 |          0 |    117120 | 35.21 |
|   LUT as Logic             | 39000 |     0 |          0 |    117120 | 33.30 |
| Block RAM Tile             |  60.5 |     0 |          0 |       144 | 42.01 |
| DSPs                       |   772 |     0 |          0 |      1248 | 61.86 |
"""
POWER = """
+--------------------------+--------------+
| Total On-Chip Power (W)  | 3.912        |
| Dynamic (W)              | 3.402        |
| Device Static (W)        | 0.510        |
+--------------------------+--------------+
| PS8                      |     2.250 |       11 |       --- |             --- |
"""


def test_p99_nearest_rank():
    assert br.p99(range(1, 1001)) == 990
    assert br.p99([5.0]) == 5.0


def test_read_reports(tmp_path):
    (tmp_path / "u.rpt").write_text(UTIL)
    (tmp_path / "p.rpt").write_text(POWER)
    assert br.read_utilization(tmp_path / "u.rpt") == {"lut": "41234", "dsp": "772",
                                                       "bram": "60.5"}
    p = br.read_power(tmp_path / "p.rpt")
    assert p == {"total": 3.912, "dynamic": 3.402, "static": 0.510, "ps": 2.250}


def test_read_times(tmp_path):
    for k in range(2):
        (tmp_path / f"times_{k}.csv").write_text(
            "id,pre_ms,load_ms,acc_ms,post_ms\n" + f"{k},1,2,3,4\n")
    t, n = br.read_times(str(tmp_path / "times_*.csv"))
    assert n == 2 and list(t["total_ms"]) == [10.0, 10.0]


def test_update_csv_idempotent(tmp_path):
    path = tmp_path / "b.csv"
    path.write_text(",".join(br.COLUMNS) + "\n" + "autre,m,f,INT8" + "," * 11 + "\n")
    ours = [{"travail": br.OURS, "fps": "5.00"}]
    br.update_csv(path, ours)
    br.update_csv(path, ours)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == len(br.PUBLISHED) + 2
    assert [r["travail"] for r in rows].count(br.OURS) == 1
    assert rows[len(br.PUBLISHED)]["travail"] == "autre"


@pytest.mark.parametrize("row", br.PUBLISHED)
def test_published_rows_complete(row):
    assert len(row) == len(br.COLUMNS)
