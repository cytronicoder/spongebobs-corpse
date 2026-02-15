from __future__ import annotations

import numpy as np


def residual_status(residuals: np.ndarray) -> str:
    if residuals.size < 4:
        return "Insufficient residual data"
    mean = float(np.mean(residuals))
    std = float(np.std(residuals))
    if std == 0:
        return "Degenerate residuals"
    z = abs(mean / std)
    if z > 0.5:
        return "Residual bias possible"
    return "Residuals centered"
