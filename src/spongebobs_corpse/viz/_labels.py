"""Label and numeric formatting helpers for figure consistency."""

from __future__ import annotations

import math


def axis_label(quantity: str, unit: str | None = None, use_math: bool = False) -> str:
    """Return axis label in the canonical `Quantity / unit` format.

    When ``use_math=True`` the function returns a matplotlib MathText string
    (wrapped in `$...$`) and converts common identifiers (e.g. ``tau``,
    ``F_peak``, ``sqrt(h)``) and units (e.g. ``mm^0.5``, ``%``) into LaTeX
    friendly forms. This keeps plain-text output unchanged for reports but
    provides rich math rendering for figures.
    """
    if not use_math:
        return f"{quantity} / {unit}" if unit else quantity

    def _ident_to_tex(token: str) -> str:
        token = token.strip()
        greek = {
            "tau": r"\tau",
            "alpha": r"\alpha",
            "beta": r"\beta",
            "sigma": r"\sigma",
            "mu": r"\mu",
        }

        # sqrt(...) -> \sqrt{...}
        if token.startswith("sqrt(") and token.endswith(")"):
            inner = token[5:-1]
            return rf"\sqrt{{{inner}}}"

        if token in greek:
            return greek[token]

        # handle underscored identifiers like `F_peak` -> F_{\mathrm{peak}}
        if "_" in token:
            base, suffix = token.split("_", 1)
            if suffix in greek:
                return rf"{base}_{{{greek[suffix]}}}"
            return rf"{base}_{{\mathrm{{{suffix}}}}}"

        return token

    parts = quantity.strip().split()
    qty_token = parts[-1] if parts else quantity
    qty_tex = _ident_to_tex(qty_token)

    unit_tex = ""
    if unit:
        if unit == "%":
            unit_tex = r"\%"
        elif "^" in unit:
            base, exp = unit.split("^", 1)
            unit_tex = rf"\mathrm{{{base}}}^{{{exp}}}"
        else:
            unit_tex = rf"\mathrm{{{unit}}}"

    return rf"${qty_tex} ({unit_tex})$" if unit_tex else rf"${qty_tex}$"


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
