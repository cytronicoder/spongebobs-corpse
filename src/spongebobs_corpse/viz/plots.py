from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..stats import confidence_band_linear
from ._labels import axis_label
from .layout import draw_gutter_legend, draw_gutter_text
from .style import apply_style, get_palette


def draw_model_panel(ax, fit: dict, x_label: str, y_label: str, title: str, color: str) -> None:
    x_plot = fit["x"]
    x_fit = np.linspace(float(np.min(x_plot)) * 0.9, float(np.max(x_plot)) * 1.1, 120)
    y_fit = fit["slope"] * x_fit + fit["intercept"]
    ci_lower, ci_upper = confidence_band_linear(x_fit, fit, level=0.95)

    ax.errorbar(
        x_plot,
        fit["y"],
        yerr=fit["yerr"],
        fmt="o",
        markersize=5.5,
        linewidth=0,
        capsize=3,
        label="Data",
        color=color,
        ecolor="black",
        markeredgecolor="black",
        markeredgewidth=0.7,
        elinewidth=0.9,
        zorder=3,
    )
    ax.plot(x_fit, y_fit, "-", color=color, linewidth=2.2, label="Model", zorder=2)
    ax.fill_between(x_fit, ci_lower, ci_upper, color=color, alpha=0.14, label="95% CI", zorder=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)


def _plot_cv_on_ax(ax, summary_df):
    cv_values = summary_df["duration_cv"].to_numpy(dtype=float) * 100
    thicknesses = summary_df["thickness_mm"].to_numpy(dtype=float)
    colors = ["#D55E00" if cv > 10 else "#F0E442" if cv > 5 else "#029E73" for cv in cv_values]
    bars = ax.bar(thicknesses, cv_values, width=8, alpha=0.8, edgecolor="black", linewidth=1.2, color=colors)
    for bar, cv in zip(bars, cv_values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{cv:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.axhline(5, color="#D55E00", linestyle="--", linewidth=1.8, alpha=0.6, label="5% threshold")
    ax.set_xlabel(axis_label("Pad thickness h", "mm"))
    ax.set_ylabel(axis_label("Coefficient of variation CV", "%"))
    ax.set_title("Experimental repeatability")
    ax.grid(True, alpha=0.2, axis="y")
    return cv_values


def draw_cv_plot(summary_df):
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    cv_values = _plot_cv_on_ax(ax, summary_df)
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(cv_values) * 1.2)
    return fig


def draw_residual_plots(params_list: list[dict]):
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    names = ["Peak Force - Linear", "Contact Duration - Linear", "Peak Force - Power-Law", "Contact Duration - Power-Law"]
    for ax, params, name in zip(axes.flat, params_list, names):
        fitted = params["fitted_values"]
        residuals = params["residuals"]
        ax.scatter(fitted, residuals, alpha=0.7, s=32, edgecolor="black")
        ax.axhline(y=0, color="#b23a48", linestyle="--", linewidth=1.2)
        ax.set_xlabel(axis_label("Fitted value", None))
        ax.set_ylabel(axis_label("Residual", None))
        ax.set_title(f"Residuals: {name}")
        ax.grid(True, alpha=0.2, axis="y")
    return fig


def draw_full_model_figure(model_specs: list[tuple], summary_lines: list[str], cv_data=None):
    apply_style()
    palette = get_palette()

    if cv_data is not None:
        nrows = 3
        height_ratios = [1.0, 1.0, 0.8]
        figsize = (15, 14)
    else:
        nrows = 2
        height_ratios = [1.0, 1.0]
        figsize = (15, 10)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(nrows, 3, width_ratios=[1.0, 1.0, 0.95], height_ratios=height_ratios)

    axes = np.array([
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
    ])
    ax_gutter = fig.add_subplot(gs[0:2, 2])
    ax_gutter.axis("off")

    for ax, spec in zip(axes.flat, model_specs):
        fit, x_label, y_label, title, color_name = spec
        draw_model_panel(ax, fit, x_label, y_label, title, palette[color_name])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = []
    for target in ("Data", "Model", "95% CI"):
        for h, lbl in zip(handles, labels):
            if lbl == target:
                order.append((h, lbl))
                break
    if order:
        draw_gutter_legend(ax_gutter, [x[0] for x in order], [x[1] for x in order], y_anchor=0.52)

    draw_gutter_text(ax_gutter, summary_lines, title="Fit summary", y_start=0.98, line_step=0.07)

    if cv_data is not None:
        ax_cv = fig.add_subplot(gs[2, :])
        cv_vals = _plot_cv_on_ax(ax_cv, cv_data)
        ax_cv.legend(loc="upper left")
        ax_cv.set_ylim(0, max(cv_vals) * 1.2)

    return fig
