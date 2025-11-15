"""
Analyze contact duration and peak force data from experimental runs.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import utils


def plot_duration_subplot(ax, detailed_df, summary_df, colors):
    """Plot duration vs thickness subplot with regression."""
    np.random.seed(42)
    for thickness in sorted(detailed_df["thickness_mm"].unique()):
        thickness_data = detailed_df[detailed_df["thickness_mm"] == thickness]
        jitter = np.random.normal(0, 0.3, len(thickness_data))
        ax.scatter(
            thickness + jitter,
            thickness_data["duration_s"],
            alpha=0.6,
            s=100,
            color=colors["gray"],
            edgecolors="black",
            linewidth=0.8,
            label=(
                "Individual runs"
                if thickness == sorted(detailed_df["thickness_mm"].unique())[0]
                else ""
            ),
            zorder=2,
        )

    x = summary_df["thickness_mm"].values
    y = summary_df["duration_s_mean"].values
    yerr = summary_df["duration_s_std"].values

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="D",
        markersize=14,
        linewidth=0,
        capsize=10,
        capthick=3,
        label="Mean ± SD",
        color=colors["blue"],
        ecolor=colors["blue"],
        markeredgecolor="black",
        markeredgewidth=1.5,
        zorder=3,
    )

    (
        slope,
        intercept,
        r_value,
        p_value,
        std_err,
        slope_uncertainty,
        intercept_uncertainty,
    ) = utils.perform_linear_regression_with_uncertainty(x, y, yerr)

    if slope is not None:
        x_fit = np.linspace(x.min() * 0.95, x.max() * 1.05, 100)
        y_fit = slope * x_fit + intercept

        n = len(x)
        t_val = stats.t.ppf(0.975, n - 2)
        x_mean = np.mean(x)
        sxx = np.sum((x - x_mean) ** 2)
        se_fit = std_err * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / sxx)
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

        # Plot uncertainty lines
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        yerr_sorted = yerr[sort_idx]

        x1_min, y1_min = x_sorted[0], y_sorted[0] + yerr_sorted[0]
        x2_min, y2_min = x_sorted[-1], y_sorted[-1] - yerr_sorted[-1]
        slope_min = (y2_min - y1_min) / (x2_min - x1_min)
        intercept_min = y1_min - slope_min * x1_min
        y_fit_min = slope_min * x_fit + intercept_min

        x1_max, y1_max = x_sorted[0], y_sorted[0] - yerr_sorted[0]
        x2_max, y2_max = x_sorted[-1], y_sorted[-1] + yerr_sorted[-1]
        slope_max = (y2_max - y1_max) / (x2_max - x1_max)
        intercept_max = y1_max - slope_max * x1_max
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

    textstr = create_regression_text(
        slope,
        intercept,
        r_value,
        p_value,
        slope_uncertainty,
        intercept_uncertainty,
        "duration",
    )

    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "black",
        },
        fontsize=12,
        family="monospace",
    )

    ax.set_xlabel("Pad Thickness ($h$, mm)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Contact Duration ($\\tau$, s)", fontsize=16, fontweight="bold")
    ax.set_title(
        "Contact Duration ($\\tau$) vs. Thickness ($h$)",
        fontsize=21,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.8)
    ax.legend(loc="best", fontsize=14, framealpha=0.95, edgecolor="black")
    ax.tick_params(labelsize=16)


def create_regression_text(
    slope,
    intercept,
    r_value,
    p_value,
    slope_uncertainty,
    intercept_uncertainty,
    variable,
):
    """Create text string for regression results."""
    if slope is None:
        return f"Insufficient data for {variable} regression analysis"

    if variable == "duration":
        eq = "$\\tau = m \\cdot h + c$"
        units_slope = "\\mathrm{s/mm}"
        units_intercept = "\\mathrm{s}"
    else:  # force
        eq = "$F = m \\cdot h + c$"
        units_slope = "\\mathrm{N/mm}"
        units_intercept = "\\mathrm{N}"

    textstr = f"Equation: {eq}\n"
    textstr += (
        f"$m = {slope:.6f} \\pm "
        f"{utils.format_uncertainty(slope_uncertainty)} \\, {units_slope}$\n"
    )
    textstr += (
        f"$c = {intercept:.4f} \\pm "
        f"{utils.format_uncertainty(intercept_uncertainty)} \\, {units_intercept}$\n"
    )
    textstr += f"$R^2 = {r_value**2:.4f}$\t"
    textstr += f"$p = {p_value:.4f}$"
    if p_value < 0.05:
        textstr += " *"
    return textstr


def plot_force_subplot(ax, summary_df, colors):
    """Plot force vs thickness subplot with regression."""
    x_force = summary_df["thickness_mm"].values
    y_force = summary_df["peak_force_N_mean"].values
    yerr_force = summary_df["peak_force_N_std"].values

    ax.errorbar(
        x_force,
        y_force,
        yerr=yerr_force,
        fmt="o",
        markersize=12,
        linewidth=0,
        capsize=8,
        capthick=2.5,
        color=colors["orange"],
        ecolor=colors["orange"],
        markeredgecolor="black",
        markeredgewidth=1.5,
        label="Mean ± SD",
        zorder=3,
    )

    (
        slope_force,
        intercept_force,
        r_value_force,
        p_value_force,
        _,
        slope_force_uncertainty,
        intercept_force_uncertainty,
    ) = utils.perform_linear_regression_with_uncertainty(x_force, y_force, yerr_force)

    if slope_force is not None:
        x_fit_force = np.linspace(x_force.min() * 0.95, x_force.max() * 1.05, 100)
        y_fit_force = slope_force * x_fit_force + intercept_force

        ax.plot(
            x_fit_force,
            y_fit_force,
            "-",
            color=colors["blue"],
            linewidth=2.5,
            label="Best fit",
            alpha=0.8,
            zorder=1,
        )

        # Plot uncertainty lines for force
        sort_idx = np.argsort(x_force)
        x_sorted = x_force[sort_idx]
        y_sorted = y_force[sort_idx]
        yerr_sorted = yerr_force[sort_idx]

        x1_min, y1_min = x_sorted[0], y_sorted[0] + yerr_sorted[0]
        x2_min, y2_min = x_sorted[-1], y_sorted[-1] - yerr_sorted[-1]
        slope_min = (y2_min - y1_min) / (x2_min - x1_min)
        intercept_min = y1_min - slope_min * x1_min
        y_fit_min = slope_min * x_fit_force + intercept_min

        x1_max, y1_max = x_sorted[0], y_sorted[0] - yerr_sorted[0]
        x2_max, y2_max = x_sorted[-1], y_sorted[-1] + yerr_sorted[-1]
        slope_max = (y2_max - y1_max) / (x2_max - x1_max)
        intercept_max = y1_max - slope_max * x1_max
        y_fit_max = slope_max * x_fit_force + intercept_max

        ax.plot(
            x_fit_force,
            y_fit_min,
            ":",
            color=colors["green"],
            linewidth=2.5,
            alpha=0.8,
            zorder=1,
        )
        ax.plot(
            x_fit_force,
            y_fit_max,
            ":",
            color=colors["green"],
            linewidth=2.5,
            alpha=0.8,
            label="Min slope",
            zorder=1,
        )
        ax.plot(
            x_fit_force,
            y_fit_max,
            ":",
            color=colors["green"],
            linewidth=2.5,
            alpha=0.8,
            label="Max slope",
            zorder=1,
        )

    textstr_force = create_regression_text(
        slope_force,
        intercept_force,
        r_value_force,
        p_value_force,
        slope_force_uncertainty,
        intercept_force_uncertainty,
        "force",
    )

    ax.text(
        0.05,
        0.05,
        textstr_force,
        transform=ax.transAxes,
        verticalalignment="bottom",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "black",
        },
        fontsize=12,
        family="monospace",
    )

    ax.set_xlabel("Pad Thickness ($h$, mm)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Peak Force ($F$, N)", fontsize=16, fontweight="bold")
    ax.set_title(
        "Peak Force ($F$) vs. Thickness ($h$)", fontsize=21, fontweight="bold", pad=15
    )
    ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.8)
    ax.legend(loc="best", fontsize=14, framealpha=0.95, edgecolor="black")
    ax.tick_params(labelsize=16)


def plot_cv_subplot(ax, summary_df, colors):
    """Plot coefficient of variation bar chart."""
    cv_values = summary_df["duration_cv"].values * 100
    x_cv = summary_df["thickness_mm"].values

    bars = ax.bar(x_cv, cv_values, width=8, alpha=0.75, edgecolor="black", linewidth=2)

    for bar_rect, cv in zip(bars, cv_values):
        if cv > 10:
            bar_rect.set_color(colors["red"])
        elif cv > 5:
            bar_rect.set_color(colors["yellow"])
        else:
            bar_rect.set_color(colors["green"])

    for bar_rect, cv in zip(bars, cv_values):
        height = bar_rect.get_height()
        ax.text(
            bar_rect.get_x() + bar_rect.get_width() / 2.0,
            height,
            f"{cv:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlabel("Pad Thickness ($h$, mm)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Coefficient of Variation ($CV$, %)", fontsize=16, fontweight="bold")
    ax.set_title(
        "Experimental Repeatability ($CV$)", fontsize=21, fontweight="bold", pad=15
    )
    ax.axhline(
        5,
        color=colors["red"],
        linestyle="--",
        linewidth=2,
        alpha=0.6,
        label="5% threshold",
    )
    ax.grid(True, alpha=0.35, axis="y", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=14, framealpha=0.95, edgecolor="black")
    ax.tick_params(labelsize=16)
    ax.set_ylim(0, max(cv_values.max(), 4.75) * 1.2)


def create_summary_table(summary_csv: Path) -> pd.DataFrame:
    """Create a formatted summary table from CSV data."""
    df = pd.read_csv(summary_csv)

    table = pd.DataFrame(
        {
            "Thickness (mm)": df["thickness_mm"],
            "Mean Duration (s)": df["duration_s_mean"].round(3),
            "Std Dev (s)": df["duration_s_std"].round(4),
            "Min (s)": df["duration_s_min"].round(3),
            "Max (s)": df["duration_s_max"].round(3),
            "N Runs": df["duration_s_count"].astype(int),
            "CV (%)": (df["duration_cv"] * 100).round(2),
            "Peak Force (N)": df["peak_force_N_mean"].round(2),
        }
    )

    return table.sort_values("Thickness (mm)")


def create_comparison_plots(detailed_csv: Path, summary_csv: Path, output_dir: Path):
    """Create comprehensive comparison plots for contact duration and force data."""
    detailed_df = pd.read_csv(detailed_csv)
    summary_df = pd.read_csv(summary_csv)

    detailed_df = detailed_df[detailed_df["thickness_mm"].notna()]
    summary_df = summary_df[summary_df["thickness_mm"].notna()]

    cb_orange = "#E69F00"
    cb_blue = "#0173B2"
    cb_green = "#029E73"
    cb_pink = "#CC78BC"
    cb_gray = "#949494"
    cb_yellow = "#F0E442"
    cb_red = "#D55E00"

    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(
        1, 3, hspace=0.25, wspace=0.2, left=0.06, right=0.97, top=0.88, bottom=0.15
    )

    ax1 = fig.add_subplot(gs[0, 0])
    plot_duration_subplot(
        ax1,
        detailed_df,
        summary_df,
        {
            "orange": cb_orange,
            "blue": cb_blue,
            "green": cb_green,
            "pink": cb_pink,
            "gray": cb_gray,
            "yellow": cb_yellow,
            "red": cb_red,
        },
    )

    ax2 = fig.add_subplot(gs[0, 1])
    plot_force_subplot(
        ax2,
        summary_df,
        {
            "orange": cb_orange,
            "blue": cb_blue,
            "green": cb_green,
            "pink": cb_pink,
            "gray": cb_gray,
            "yellow": cb_yellow,
            "red": cb_red,
        },
    )

    ax3 = fig.add_subplot(gs[0, 2])
    plot_cv_subplot(
        ax3,
        summary_df,
        {
            "orange": cb_orange,
            "blue": cb_blue,
            "green": cb_green,
            "pink": cb_pink,
            "gray": cb_gray,
            "yellow": cb_yellow,
            "red": cb_red,
        },
    )

    output_path = output_dir / "comprehensive_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def create_individual_comparison(detailed_csv: Path, output_dir: Path):
    """
    Create individual run comparison across thicknesses.
    """
    detailed_df = pd.read_csv(detailed_csv)
    detailed_df = detailed_df[detailed_df["thickness_mm"].notna()]

    cb_colors = ["#E69F00", "#0173B2", "#029E73", "#CC78BC", "#F0E442", "#D55E00"]

    _, ax = plt.subplots(figsize=(14, 7))

    runs = detailed_df["run"].unique()
    thickness_values = sorted(detailed_df["thickness_mm"].unique())

    x = np.arange(len(thickness_values))
    width = 0.15

    for i, run in enumerate(runs):
        run_data = detailed_df[detailed_df["run"] == run].sort_values("thickness_mm")
        if len(run_data) > 0:
            durations = [
                (
                    run_data[run_data["thickness_mm"] == t]["duration_s"].values[0]
                    if len(run_data[run_data["thickness_mm"] == t]) > 0
                    else 0
                )
                for t in thickness_values
            ]
            color = cb_colors[i % len(cb_colors)]
            ax.bar(
                x + i * width,
                durations,
                width,
                label=run,
                alpha=0.85,
                color=color,
                edgecolor="black",
                linewidth=1.2,
            )

    ax.set_xlabel("Pad Thickness (mm)", fontsize=16, fontweight="bold")
    ax.set_ylabel("Contact Duration (s)", fontsize=16, fontweight="bold")
    ax.set_title(
        "Contact Duration by Run and Thickness", fontsize=21, fontweight="bold", pad=15
    )
    ax.set_xticks(x + width * (len(runs) - 1) / 2)
    ax.set_xticklabels([f"{t:.0f}" for t in thickness_values], fontsize=14)
    ax.legend(
        title="Run",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=14,
        title_fontsize=14,
        framealpha=0.95,
        edgecolor="black",
    )
    ax.grid(True, alpha=0.35, axis="y", linestyle="--", linewidth=0.8)
    ax.tick_params(labelsize=16)

    plt.tight_layout(pad=1.5)
    output_path = output_dir / "runs_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return output_path


def write_report_header(f, summary_df, detailed_df):
    """Write the header section of the report."""
    f.write("=" * 80 + "\n")
    f.write("CONTACT DURATION ANALYSIS - STATISTICAL REPORT\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Measurements: {len(detailed_df)}\n")
    f.write(f"Thickness Conditions: {len(summary_df)}\n")
    f.write(f"Runs per Condition: {int(summary_df['duration_s_count'].iloc[0])}\n\n")


def write_summary_statistics(f, summary_csv):
    """Write the summary statistics section."""
    f.write("-" * 80 + "\n")
    f.write("SUMMARY STATISTICS BY PAD THICKNESS\n")
    f.write("-" * 80 + "\n\n")

    table = create_summary_table(summary_csv.parent / summary_csv.name)
    f.write(table.to_string(index=False))
    f.write("\n\n")


def write_statistical_tests(f, summary_df, detailed_df):
    """Write the statistical tests section."""
    f.write("-" * 80 + "\n")
    f.write("STATISTICAL TESTS\n")
    f.write("-" * 80 + "\n\n")

    x = summary_df["thickness_mm"].values
    y = summary_df["duration_s_mean"].values

    if len(x) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        f.write("Linear Regression Analysis:\n")
        f.write(f"  Equation: Duration = {slope:.6f} × Thickness + {intercept:.6f}\n")
        f.write(f"  R² = {r_value**2:.4f}\n")
        f.write(f"  p-value = {p_value:.4f}\n")
        f.write(f"  Standard Error = {std_err:.6f}\n\n")

        if p_value < 0.05:
            f.write(
                "  ✓ SIGNIFICANT: Statistically significant relationship (p < 0.05)\n"
            )
        else:
            f.write("  ✗ NOT SIGNIFICANT: No significant relationship (p ≥ 0.05)\n")
        f.write("\n")

    if len(summary_df) >= 3:
        groups = [
            detailed_df[detailed_df["thickness_mm"] == t]["duration_s"].values
            for t in sorted(detailed_df["thickness_mm"].unique())
        ]

        f_stat, p_anova = stats.f_oneway(*groups)
        f.write("One-Way ANOVA:\n")
        f.write(f"  F-statistic = {f_stat:.4f}\n")
        f.write(f"  p-value = {p_anova:.4f}\n")

        if p_anova < 0.05:
            f.write(
                "  ✓ SIGNIFICANT: At least one thickness differs significantly (p < 0.05)\n"
            )
        else:
            f.write(
                "  ✗ NOT SIGNIFICANT: No significant difference between groups (p ≥ 0.05)\n"
            )
        f.write("\n")


def write_detailed_measurements(f, detailed_df):
    """Write the detailed measurements section."""
    f.write("-" * 80 + "\n")
    f.write("DETAILED MEASUREMENTS\n")
    f.write("-" * 80 + "\n\n")

    for thickness in sorted(detailed_df["thickness_mm"].unique()):
        f.write(f"\nThickness: {thickness:.0f} mm\n")
        f.write("-" * 40 + "\n")

        thickness_data = detailed_df[detailed_df["thickness_mm"] == thickness]

        for _, row in thickness_data.iterrows():
            f.write(f"  {row['run']}: {row['duration_s']:.3f} s ")
            f.write(f"(Peak: {row['peak_force_N']:.2f} N, ")
            f.write(f"Start: {row['start_time_s']:.2f} s, ")
            f.write(f"End: {row['end_time_s']:.2f} s)\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 80 + "\n")


def generate_text_report(summary_csv: Path, detailed_csv: Path, output_dir: Path):
    """Generate a detailed statistical report in text format."""
    summary_df = pd.read_csv(summary_csv)
    detailed_df = pd.read_csv(detailed_csv)

    summary_df = summary_df[summary_df["thickness_mm"].notna()]
    detailed_df = detailed_df[detailed_df["thickness_mm"].notna()]

    report_path = output_dir / "statistical_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        write_report_header(f, summary_df, detailed_df)
        write_summary_statistics(f, summary_csv)
        write_statistical_tests(f, summary_df, detailed_df)
        write_detailed_measurements(f, detailed_df)

    return report_path


def main():
    """Main function to run the analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive summary report with visualizations and statistics."
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default="outputs/contact_analysis_10pct",
        help="Directory containing analysis results (default: outputs/contact_analysis_10pct)",
    )

    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)

    if not analysis_dir.exists():
        print(f"Error: Analysis directory not found: {analysis_dir}")
        print("Please run contact_duration.py first to generate analysis results.")
        return

    summary_csv = analysis_dir / "contact_duration_summary.csv"
    detailed_csv = analysis_dir / "contact_duration_detailed.csv"

    if not summary_csv.exists() or not detailed_csv.exists():
        print("Error: Required CSV files not found.")
        print(f"  Looking for: {summary_csv}")
        print(f"  Looking for: {detailed_csv}")
        return

    summary_table = create_summary_table(summary_csv)

    table_path = analysis_dir / "summary_table_formatted.csv"
    summary_table.to_csv(table_path, index=False)

    create_comparison_plots(detailed_csv, summary_csv, analysis_dir)

    create_individual_comparison(detailed_csv, analysis_dir)

    generate_text_report(summary_csv, detailed_csv, analysis_dir)


if __name__ == "__main__":
    main()
