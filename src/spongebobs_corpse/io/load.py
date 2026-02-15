from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("h", "Run", "tau", "F_peak")


def load_batch_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    data = pd.DataFrame(
        {
            "thickness_mm": frame["h"].astype(float),
            "run": frame["Run"].map(lambda val: f"Run {int(val)}"),
            "duration_s": frame["tau"].astype(float),
            "peak_force_N": frame["F_peak"].astype(float),
            "method": "threshold",
        }
    )
    return data
