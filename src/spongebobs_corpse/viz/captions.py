from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .labels import format_p_value, format_value_uncertainty


def _build_model_line(model: Mapping[str, Any]) -> str:
    equation = model.get("equation", "Model not specified")
    params = model.get("parameters", {})
    parts = []
    for key, value in params.items():
        unc = model.get("uncertainties", {}).get(key)
        parts.append(f"{key}={format_value_uncertainty(value, unc)}")
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
