"""Reusable figure constructors and gutter utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt


def make_single_panel(figsize: tuple[float, float] = (7.0, 4.6)):
    """Create a single-panel figure with constrained layout."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return fig, ax


def make_panel_with_gutter(
    width_ratio: tuple[float, float] = (3.5, 1.5),
    figsize: tuple[float, float] = (10.0, 5.0),
):
    """Create a main panel plus right-side gutter for legend and fit summary."""
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=width_ratio)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_gutter = fig.add_subplot(gs[0, 1])
    ax_gutter.axis("off")
    return fig, ax_main, ax_gutter


def make_multipanel(
    nrows: int,
    ncols: int,
    figsize: tuple[float, float] = (12.0, 8.0),
):
    """Create a multipanel figure with consistent layout behavior."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    return fig, axes


def draw_gutter_text(
    ax_gutter,
    lines: Sequence[str],
    title: str | None = None,
    y_start: float = 0.98,
    line_step: float = 0.09,
) -> None:
    """Draw left-aligned text block in the gutter axis."""
    y = y_start
    if title:
        ax_gutter.text(0.0, y, title, ha="left", va="top", fontsize=11, fontweight="bold")
        y -= line_step * 1.2

    for line in lines:
        ax_gutter.text(0.0, y, line, ha="left", va="top", fontsize=10)
        y -= line_step


def draw_gutter_legend(ax_gutter, handles: Iterable, labels: Iterable[str], y_anchor: float = 0.55):
    """Place a legend block in the gutter."""
    legend = ax_gutter.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.0, y_anchor),
        frameon=True,
        borderaxespad=0.0,
    )
    return legend
