"""T19.6: number format of the generated tables (decimal point, comma thousands)."""

import numpy as np

from yolo import fmt


def test_num_and_pct():
    assert fmt.num(132031) == "132,031" and fmt.num(np.int64(4952)) == "4,952"
    assert fmt.num(56.3, 2) == "56.30" and fmt.num(1234.56, 1) == "1,234.6"
    assert fmt.num(None) == fmt.MISSING == "—"
    assert fmt.pct(0.443) == "44.3 %" and fmt.pct(0.2583, 2) == "25.83 %"
    assert fmt.pct(None) == "—"
