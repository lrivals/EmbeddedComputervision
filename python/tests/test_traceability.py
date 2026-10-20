"""T28.3: every symbol of the traceability table exists; a missing one is reported."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("traceability", ROOT / "tools" / "traceability.py")
tr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tr)


def test_every_symbol_found():
    _, missing = tr.render()
    assert missing == []


def test_missing_symbol_detected(monkeypatch):
    bad = ("[fake]", "—", [f"{tr.PY_INT}::no_such_function"], [f"{tr.GOLDEN_CONV}::requantize"],
           [f"{tr.HLS_OUT}::store_tile"], ["no_such_field"])
    monkeypatch.setattr(tr, "TABLE", tr.TABLE + [bad])
    text, missing = tr.render()
    assert f"{tr.PY_INT}::no_such_function" in missing
    assert f"{tr.DESC}::no_such_field" in missing
    assert "MISSING" in text


def test_lines_point_at_definitions():
    path, line = tr.locate(f"{tr.GOLDEN_CONV}::leaky_int")
    assert "leaky_int(" in (ROOT / path).read_text().splitlines()[line - 1]
    path, line = tr.locate(f"{tr.PY_MODEL}::IntNetwork.forward")
    assert "def forward" in (ROOT / path).read_text().splitlines()[line - 1]


def test_committed_doc_up_to_date():
    text, _ = tr.render()
    assert tr.DOC.read_text() == text, "python tools/traceability.py"
