from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot bootstrap empty input")
    samples = rng.choice(values, size=(n_boot, values.size), replace=True)
    means = np.mean(samples, axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
