from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FitSummary:
    parameter: str
    model: str
    slope: float
    slope_se: float
    intercept: float
    intercept_se: float
    r2: float
    adj_r2: float
    p_value_slope: float
    p_value_intercept: float
