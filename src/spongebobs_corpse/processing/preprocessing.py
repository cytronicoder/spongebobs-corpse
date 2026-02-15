from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("thickness_mm", "run", "duration_s", "peak_force_N", "method")


def validate_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    out = frame.copy()
    for col in ("thickness_mm", "duration_s", "peak_force_N"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    subset = ["thickness_mm", "duration_s", "peak_force_N"]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=subset)
    return out
