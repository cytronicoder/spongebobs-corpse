from __future__ import annotations

import numpy as np


def combine_uncertainty(random_component: np.ndarray, instrument: float) -> np.ndarray:
    return np.sqrt(np.square(random_component) + instrument**2)
