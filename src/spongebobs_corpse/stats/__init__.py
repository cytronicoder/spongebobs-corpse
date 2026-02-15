from .bootstrap import bootstrap_mean_ci
from .hypothesis import pairwise_adjacent_comparisons
from .regression import (
    adjacent_thickness_comparisons,
    bonferroni_correction,
    confidence_band_linear,
    fit_linear,
    fit_powerlaw,
    format_p_value,
)

__all__ = [
    "fit_linear",
    "fit_powerlaw",
    "confidence_band_linear",
    "adjacent_thickness_comparisons",
    "bonferroni_correction",
    "format_p_value",
    "pairwise_adjacent_comparisons",
    "bootstrap_mean_ci",
]
