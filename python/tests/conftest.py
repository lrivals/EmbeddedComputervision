"""Avec YOLO_REQUIRE_MODEL=1 (CI, T10.13), un test sauté faute d'export ou de binaire C++
échoue : la CI génère ces entrées (`make ci-model`, `make ci`), elles doivent être là.

Les sauts liés aux poids Darknet, à VOC ou à une option facultative restent permis.
"""

import os

import pytest

REQUIRED = ("export absent", "non exporté", "golden_run", "tb_stream", "hls-cycles")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if not (rep.skipped and os.environ.get("YOLO_REQUIRE_MODEL")):
        return
    reason = str(rep.longrepr[2]) if isinstance(rep.longrepr, tuple) else str(rep.longrepr)
    if any(k in reason for k in REQUIRED):
        rep.outcome = "failed"
        rep.longrepr = f"saut interdit par YOLO_REQUIRE_MODEL : {reason}"
