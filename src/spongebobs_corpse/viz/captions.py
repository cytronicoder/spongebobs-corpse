from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .labels import format_p_value, format_value_uncertainty


def _build_model_line(model: Mapping[str, Any]) -> str:
    equation = model.get("equation", "Model not specified")
    # convert simple plain-text equations to MathText-friendly LaTeX
    if isinstance(equation, str) and not equation.strip().startswith("$"):
        eq_tex = equation.replace("*", " ")
        if "sqrt(" in eq_tex:
            eq_tex = eq_tex.replace("sqrt(", "\\sqrt{").replace(")", "}")
        equation = f"${eq_tex}$"

    params = model.get("parameters", {})
    parts = []

    def _key_to_tex(key: str) -> str:
        if "_" in key:
            base, suffix = key.split("_", 1)
            greek = {"tau": r"\\tau"}
            if suffix in greek:
                return rf"{base}_{{{greek[suffix]}}}"
            return rf"{base}_{{\\mathrm{{{suffix}}}}}"
        return key

    for key, value in params.items():
        unc = model.get("uncertainties", {}).get(key)
        key_tex = _key_to_tex(key)
        parts.append(f"${key_tex}={format_value_uncertainty(value, unc)}$")

    suffix = f" ({'; '.join(parts)})" if parts else ""
    return f"Model: {equation}{suffix}"


def build_caption(metadata: Mapping[str, Any]) -> str:
    lines: list[str] = []
    if metadata.get("data"):
        lines.append(f"Data: {metadata['data']}")
    if metadata.get("uncertainty"):
        lines.append(f"Uncertainty: {metadata['uncertainty']}")

    model = metadata.get("model")
    if isinstance(model, Mapping):
        lines.append(_build_model_line(model))

    fit = metadata.get("fit")
    if isinstance(fit, Mapping):
        bits = []
        if "r2" in fit:
            bits.append(f"R^2={fit['r2']:.4f}")
        if "adj_r2" in fit:
            bits.append(f"adj R^2={fit['adj_r2']:.4f}")
        if "p_value" in fit:
            bits.append(format_p_value(float(fit["p_value"])))
        if bits:
            lines.append("Fit stats: " + ", ".join(bits))

    tail = []
    if metadata.get("n") is not None:
        tail.append(f"n={metadata['n']}")
    if metadata.get("method"):
        tail.append(f"fit method={metadata['method']}")
    if tail:
        lines.append("; ".join(tail))

    if metadata.get("note"):
        lines.append(str(metadata["note"]))

    return "\n".join(lines[:7])
