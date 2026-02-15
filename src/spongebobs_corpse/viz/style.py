"""Central matplotlib style for all figures."""

from __future__ import annotations

import matplotlib as mpl

PALETTE: dict[str, str] = {
    "blue": "#1f4e79",
    "orange": "#c05a00",
    "green": "#2d7f5e",
    "red": "#b23a48",
    "purple": "#6a4c93",
    "gray": "#7f7f7f",
    "black": "#1a1a1a",
}


def apply_style() -> None:
    """Apply global plotting style for publication-ready consistency."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.2,
            "lines.markersize": 6.5,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.grid": False,
            "axes.formatter.use_mathtext": True,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#d0d0d0",
        }
    )


def get_palette() -> dict[str, str]:
    """Return the shared color palette."""
    return dict(PALETTE)
