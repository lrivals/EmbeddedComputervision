"""Number format of the generated tables and text (M19, T19.6): decimal point, comma as the
thousands separator (56.30, 132,031), `—` for a missing value."""

import numpy as np

MISSING = "—"


def num(x, d=1):
    """`x` with `d` decimals (an integer without any), thousands separated by commas."""
    if x is None:
        return MISSING
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{x:,.{d}f}"


def pct(x, d=1):
    """Share `x` (0-1) as a percentage: `0.443` → `44.3 %`."""
    return MISSING if x is None else f"{num(100 * x, d)} %"
