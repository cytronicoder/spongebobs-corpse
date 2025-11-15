"""
Shared utilities for data analysis scripts.
"""

import numpy as np
from scipy import stats


def format_uncertainty(value):
    """Format uncertainty to 1 sf (or 2 sf if starting with 1)"""
    if value is None or value == 0 or np.isnan(value):
        return "N/A"
    magnitude = 10 ** np.floor(np.log10(abs(value)))
    first_digit = int(abs(value) / magnitude)
    if first_digit == 1:
        formatted = f"{value:.2e}" if value < 0.01 else f"{value:.3f}"
    else:
        formatted = f"{value:.1e}" if value < 0.01 else f"{value:.2f}"

    if "e" in formatted:
        mantissa, exp_part = formatted.split("e")
        exp = int(exp_part)
        return f"{mantissa} \\times 10^{{{exp}}}"
    return formatted


def perform_linear_regression_with_uncertainty(x, y, yerr):
    """Perform linear regression and calculate slope/intercept uncertainties via bootstrap."""
    if len(x) < 2:
        return None, None, None, None, None, None, None

    # Basic regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Bootstrap for uncertainties
    n_boot = 1000
    slopes = []
    intercepts = []

    for _ in range(n_boot):
        indices = np.random.choice(len(x), len(x), replace=True)
        x_boot = x[indices]
        y_boot = y[indices]
        if yerr is not None:
            yerr_boot = yerr[indices]
            # Add noise based on errors
            noise = np.random.normal(0, yerr_boot)
            y_boot += noise
        try:
            s, i, _, _, _ = stats.linregress(x_boot, y_boot)
            slopes.append(s)
            intercepts.append(i)
        except:
            continue

    if slopes:
        slope_uncertainty = np.std(slopes)
        intercept_uncertainty = np.std(intercepts)
    else:
        slope_uncertainty = intercept_uncertainty = None

    return (
        slope,
        intercept,
        r_value,
        p_value,
        std_err,
        slope_uncertainty,
        intercept_uncertainty,
    )


def calculate_slope_uncertainty_values(x, y, yerr):
    """Calculate slope and intercept uncertainties from error bars."""
    if len(x) >= 2 and yerr is not None:
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        yerr_sorted = yerr[sort_idx]

        x1_min, y1_min = x_sorted[0], y_sorted[0] + yerr_sorted[0]
        x2_min, y2_min = x_sorted[-1], y_sorted[-1] - yerr_sorted[-1]
        slope_min = (y2_min - y1_min) / (x2_min - x1_min)
        intercept_min = y1_min - slope_min * x1_min

        x1_max, y1_max = x_sorted[0], y_sorted[0] - yerr_sorted[0]
        x2_max, y2_max = x_sorted[-1], y_sorted[-1] + yerr_sorted[-1]
        slope_max = (y2_max - y1_max) / (x2_max - x1_max)
        intercept_max = y1_max - slope_max * x1_max

        slope_uncertainty = abs(slope_max - slope_min) / 2
        intercept_uncertainty = abs(intercept_max - intercept_min) / 2

        return slope_uncertainty, intercept_uncertainty
    return None, None


def plot_slope_uncertainty_bounds(ax, x, y, yerr, x_fit, colors):
    """Plot the min and max slope bounds."""
    if len(x) >= 2 and yerr is not None:
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        yerr_sorted = yerr[sort_idx]

        x1_min, y1_min = x_sorted[0], y_sorted[0] + yerr_sorted[0]
        x2_min, y2_min = x_sorted[-1], y_sorted[-1] - yerr_sorted[-1]
        slope_min = (y2_min - y1_min) / (x2_min - x1_min)
        intercept_min = y1_min - slope_min * x1_min

        x1_max, y1_max = x_sorted[0], y_sorted[0] - yerr_sorted[0]
        x2_max, y2_max = x_sorted[-1], y_sorted[-1] + yerr_sorted[-1]
        slope_max = (y2_max - y1_max) / (x2_max - x1_max)
        intercept_max = y1_max - slope_max * x1_max

        y_fit_min = slope_min * x_fit + intercept_min
        y_fit_max = slope_max * x_fit + intercept_max

        ax.plot(
            x_fit,
            y_fit_min,
            ":",
            color=colors["pink"],
            linewidth=2.5,
            alpha=0.8,
            label="Min slope",
            zorder=1,
        )
        ax.plot(
            x_fit,
            y_fit_max,
            ":",
            color=colors["pink"],
            linewidth=2.5,
            alpha=0.8,
            label="Max slope",
            zorder=1,
        )


def plot_regression_and_ci(ax, x, y, reg_stats, colors):
    """Plot regression line and confidence interval."""
    if not np.isnan(reg_stats["slope"]) and len(x) >= 2:
        x_fit = np.linspace(x.min() * 0.95, x.max() * 1.05, 100)
        y_fit = reg_stats["slope"] * x_fit + reg_stats["intercept"]

        n = len(x)
        t_val = stats.t.ppf(0.975, n - 2)
        s_err = reg_stats["std_err"]

        x_mean = np.mean(x)
        sxx = np.sum((x - x_mean) ** 2)
        se_fit = s_err * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / sxx)
        ci = t_val * se_fit

        ax.plot(
            x_fit,
            y_fit,
            "-",
            color=colors["orange"],
            linewidth=3,
            label="Best fit",
            zorder=1,
        )

        ax.fill_between(
            x_fit,
            y_fit - ci,
            y_fit + ci,
            alpha=0.2,
            color=colors["orange"],
            label="95% CI",
            zorder=0,
        )

        return x_fit
    return None


def get_errorbar_kwargs(colors):
    """Get common errorbar kwargs."""
    return {
        "fmt": "o",
        "markersize": 12,
        "capsize": 8,
        "capthick": 2.5,
        "label": "Mean ± SD",
        "color": colors["blue"],
        "ecolor": colors["blue"],
        "markeredgecolor": "black",
        "markeredgewidth": 1.5,
        "linewidth": 2.5,
        "zorder": 3,
    }


def get_text_bbox():
    """Get common text bbox."""
    return {
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.9,
        "edgecolor": "black",
    }


def sort_run_key(p):
    """Sort key for runs."""
    import re

    if p.startswith("Run "):
        m = re.match(r"Run\s+(\d+)", p)
        if m:
            return (0, int(m.group(1)), "")
        return (0, 10_000, p)
    if p == "Latest":
        return (1, 0, "")
    return (2, 0, p)
