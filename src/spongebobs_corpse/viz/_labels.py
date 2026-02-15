"""Label and numeric formatting helpers for figure consistency."""

from __future__ import annotations

import math


def axis_label(quantity: str, unit: str | None = None) -> str:
    """Return axis label in the canonical `Quantity / unit` format."""
    return f"{quantity} / {unit}" if unit else quantity


def format_value_uncertainty(value: float, uncertainty: float | None) -> str:
    """Format value ± uncertainty with uncertainty rounded to 1 significant figure."""
    if uncertainty is None or uncertainty == 0 or math.isnan(uncertainty):
        return f"{value:.4g}"

    unc = float(f"{uncertainty:.1g}")
    if unc == 0:
        return f"{value:.4g}"

    if abs(unc) >= 1:
        decimals = 0
    else:
        decimals = -int(math.floor(math.log10(abs(unc))))

    return f"{value:.{decimals}f} +/- {unc:.{decimals}f}"


def format_p_value(p_value: float) -> str:
    """Format p-value consistently for reports and captions."""
    if p_value < 1e-3:
        return "p < 0.001"
    return f"p = {p_value:.3f}"
