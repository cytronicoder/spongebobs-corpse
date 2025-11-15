"""
Analyzes contact duration from aligned force-time data.

The script implements multiple contact detection methods:
1. Force threshold method (primary)
2. Velocity-based method
3. Energy-based method
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter


def extract_thickness_from_filename(filename: str) -> Optional[float]:
    """
    Extract pad thickness in mm from filename.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*mm", filename, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def detect_runs(columns: pd.MultiIndex) -> List[str]:
    """
    Extract unique run labels from aligned DataFrame multi-index columns.
    Returns runs in sorted order (Run 1, Run 2, ..., Latest).
    """
    runs = sorted(set([col[0] for col in columns]))

    def sort_key(p: str) -> Tuple[int, int, str]:
        if p.startswith("Run "):
            m = re.match(r"Run\s+(\d+)", p)
            if m:
                return (0, int(m.group(1)), "")
            return (0, 10_000, p)
        if p == "Latest":
            return (1, 0, "")
        return (2, 0, p)

    return sorted(runs, key=sort_key)


def smooth_signal(
    data: np.ndarray, window_length: int = 11, polyorder: int = 3
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter to smooth noisy signals.

    Args:
        data: Input signal
        window_length: Length of filter window (must be odd)
        polyorder: Order of polynomial fit

    Returns:
        Smoothed signal
    """
    if len(data) < window_length:
        return data

    mask = np.isfinite(data)
    if not mask.any():
        return data

    smoothed = data.copy()
    if mask.sum() >= window_length:
        smoothed[mask] = savgol_filter(data[mask], window_length, polyorder)

    return smoothed


def calculate_contact_duration_threshold(
    time: np.ndarray,
    force: np.ndarray,
    threshold_fraction: float = 0.05,
    smooth: bool = True,
) -> Dict[str, float]:
    """
    Calculate contact duration using force threshold method.

    Contact starts when |force| first exceeds threshold_fraction * max|force|
    Contact ends when |force| drops below threshold after peak.

    Args:
        time: Time array (aligned)
        force: Force array
        threshold_fraction: Fraction of peak force to use as threshold
        smooth: Whether to apply smoothing to force signal

    Returns:
        Dictionary with contact metrics:
            - duration: Contact duration in seconds
            - start_time: Time when contact starts
            - end_time: Time when contact ends
            - peak_force: Maximum absolute force
            - threshold_force: Threshold value used
    """
    mask = np.isfinite(time) & np.isfinite(force)
    if not mask.any():
        return {
            "duration": np.nan,
            "start_time": np.nan,
            "end_time": np.nan,
            "peak_force": np.nan,
            "threshold_force": np.nan,
            "method": "threshold",
        }

    t_valid = time[mask]
    f_valid = force[mask]

    if smooth:
        f_valid = smooth_signal(f_valid)

    abs_force = np.abs(f_valid)
    peak_force = np.max(abs_force)
    threshold = threshold_fraction * peak_force

    peak_idx = np.argmax(abs_force)

    start_idx = 0
    for i in range(peak_idx, -1, -1):
        if abs_force[i] < threshold:
            start_idx = i + 1 if i + 1 < len(abs_force) else i
            break

    end_idx = len(abs_force) - 1
    for i in range(peak_idx, len(abs_force)):
        if abs_force[i] < threshold:
            end_idx = i
            break

    start_time = t_valid[start_idx]
    end_time = t_valid[end_idx]
    duration = end_time - start_time

    return {
        "duration": duration,
        "start_time": start_time,
        "end_time": end_time,
        "peak_force": peak_force,
        "threshold_force": threshold,
        "method": "threshold",
    }


def calculate_contact_duration_velocity(
    time: np.ndarray, velocity: np.ndarray, smooth: bool = True
) -> Dict[str, float]:
    """
    Calculate contact duration using velocity reversal method.

    Contact duration is the time during which velocity is negative
    (cart decelerating/reversing during impact).

    Args:
        time: Time array (aligned)
        velocity: Velocity array
        smooth: Whether to apply smoothing

    Returns:
        Dictionary with contact metrics
    """
    mask = np.isfinite(time) & np.isfinite(velocity)
    if not mask.any():
        return {
            "duration": np.nan,
            "start_time": np.nan,
            "end_time": np.nan,
            "min_velocity": np.nan,
            "method": "velocity",
        }

    t_valid = time[mask]
    v_valid = velocity[mask]

    if smooth:
        v_valid = smooth_signal(v_valid)

    min_vel_idx = np.argmin(v_valid)
    min_vel = v_valid[min_vel_idx]

    start_idx = 0
    for i in range(min_vel_idx, -1, -1):
        if v_valid[i] >= 0:
            start_idx = i + 1 if i + 1 < len(v_valid) else i
            break

    end_idx = len(v_valid) - 1
    for i in range(min_vel_idx, len(v_valid)):
        if v_valid[i] >= 0:
            end_idx = i
            break

    start_time = t_valid[start_idx]
    end_time = t_valid[end_idx]
    duration = end_time - start_time

    return {
        "duration": duration,
        "start_time": start_time,
        "end_time": end_time,
        "min_velocity": min_vel,
        "method": "velocity",
    }


def analyze_contact_duration_for_file(
    aligned_csv_path: Path,
    threshold_fraction: float = 0.05,
    smooth: bool = True,
    method: str = "threshold",
) -> pd.DataFrame:
    """
    Analyze contact duration for all runs in an aligned CSV file.

    Args:
        aligned_csv_path: Path to aligned CSV file
        threshold_fraction: Threshold for contact detection
        smooth: Whether to smooth signals
        method: Detection method ('threshold', 'velocity', or 'both')

    Returns:
        DataFrame with contact duration metrics for each run
    """
    df = pd.read_csv(aligned_csv_path, header=[0, 1])
    runs = detect_runs(df.columns)

    thickness = extract_thickness_from_filename(aligned_csv_path.stem)

    results = []

    for run in runs:
        try:
            time = df[(run, "Time (s) [aligned]")].values
            force = df[(run, "Force (N)")].values

            if method in ["threshold", "both"]:
                metrics_threshold = calculate_contact_duration_threshold(
                    time, force, threshold_fraction, smooth
                )

                result = {
                    "file": aligned_csv_path.stem,
                    "thickness_mm": thickness,
                    "run": run,
                    "duration_s": metrics_threshold["duration"],
                    "start_time_s": metrics_threshold["start_time"],
                    "end_time_s": metrics_threshold["end_time"],
                    "peak_force_N": metrics_threshold["peak_force"],
                    "threshold_force_N": metrics_threshold["threshold_force"],
                    "method": "threshold",
                }
                results.append(result)

            if method in ["velocity", "both"]:
                if (run, "Velocity (m/s)") in df.columns:
                    velocity = df[(run, "Velocity (m/s)")].values
                    metrics_velocity = calculate_contact_duration_velocity(
                        time, velocity, smooth
                    )

                    result = {
                        "file": aligned_csv_path.stem,
                        "thickness_mm": thickness,
                        "run": run,
                        "duration_s": metrics_velocity["duration"],
                        "start_time_s": metrics_velocity["start_time"],
                        "end_time_s": metrics_velocity["end_time"],
                        "min_velocity_m_s": metrics_velocity["min_velocity"],
                        "method": "velocity",
                    }
                    results.append(result)

        except Exception as e:
            print(f"Warning: Failed to process {run} in {aligned_csv_path.name}: {e}")
            continue

    return pd.DataFrame(results)


def aggregate_by_thickness(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate contact duration results by pad thickness.

    Calculates mean, std, min, max, count, and uncertainty for each thickness.
    Uncertainty is calculated as (max - min) / 2.
    """
    valid_df = results_df[results_df["duration_s"].notna()].copy()

    if len(valid_df) == 0:
        return pd.DataFrame()

    agg_funcs = {
        "duration_s": ["mean", "std", "min", "max", "count"],
        "peak_force_N": (
            ["mean", "std", "min", "max"] if "peak_force_N" in valid_df.columns else []
        ),
    }

    agg_funcs = {k: v for k, v in agg_funcs.items() if v}

    grouped = valid_df.groupby(["thickness_mm", "method"]).agg(agg_funcs)
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()

    if "duration_s_mean" in grouped.columns and "duration_s_std" in grouped.columns:
        grouped["duration_cv"] = grouped["duration_s_std"] / grouped["duration_s_mean"]

    if "duration_s_min" in grouped.columns and "duration_s_max" in grouped.columns:
        grouped["duration_s_uncertainty"] = (
            grouped["duration_s_max"] - grouped["duration_s_min"]
        ) / 2

    if "peak_force_N_min" in grouped.columns and "peak_force_N_max" in grouped.columns:
        grouped["peak_force_N_uncertainty"] = (
            grouped["peak_force_N_max"] - grouped["peak_force_N_min"]
        ) / 2

    return grouped.sort_values("thickness_mm")


def perform_regression_analysis(
    aggregated_df: pd.DataFrame, method: str = "threshold"
) -> Dict:
    """
    Perform linear regression: contact duration vs. pad thickness.

    Returns regression statistics and model parameters.
    """
    method_df = aggregated_df[aggregated_df["method"] == method].copy()

    if len(method_df) < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r_squared": np.nan,
            "p_value": np.nan,
            "std_err": np.nan,
            "n_points": len(method_df),
        }

    x = method_df["thickness_mm"].values
    y = method_df["duration_s_mean"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_value**2,
        "p_value": p_value,
        "std_err": std_err,
        "n_points": len(method_df),
        "equation": f"Duration = {slope:.6f} * Thickness + {intercept:.6f}",
    }


def plot_duration_vs_thickness(
    results_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    output_path: Path,
    method: str = "threshold",
):
    """
    Create publication-quality plot of contact duration vs. pad thickness.
    Uses colorblind-friendly palette and shows best fit with confidence bands.
    """
    method_results = results_df[results_df["method"] == method].copy()
    method_agg = aggregated_df[aggregated_df["method"] == method].copy()

    if len(method_agg) == 0:
        print(f"No data available for {method} method")
        return

    CB_ORANGE = "#E69F00"
    CB_BLUE = "#0173B2"
    CB_GREEN = "#029E73"
    CB_PINK = "#DE8F05"
    CB_GRAY = "#949494"

    fig, ax = plt.subplots(figsize=(10, 7))

    for thickness in sorted(method_results["thickness_mm"].unique()):
        if pd.isna(thickness):
            continue
        thickness_data = method_results[method_results["thickness_mm"] == thickness]
        ax.scatter(
            [thickness] * len(thickness_data),
            thickness_data["duration_s"].values,
            alpha=0.5,
            s=80,
            color=CB_GRAY,
            edgecolors="black",
            linewidth=0.5,
            label=(
                "Individual runs"
                if thickness == sorted(method_results["thickness_mm"].unique())[0]
                else ""
            ),
            zorder=2,
        )

    x = method_agg["thickness_mm"].values
    y = method_agg["duration_s_mean"].values
    yerr = (
        method_agg["duration_s_std"].values
        if "duration_s_std" in method_agg.columns
        else None
    )

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        markersize=12,
        capsize=8,
        capthick=2.5,
        label="Mean ± SD",
        color=CB_BLUE,
        ecolor=CB_BLUE,
        markeredgecolor="black",
        markeredgewidth=1.5,
        linewidth=2.5,
        zorder=3,
    )

    reg_stats = perform_regression_analysis(aggregated_df, method)

    if not np.isnan(reg_stats["slope"]) and len(x) >= 2:
        from scipy import stats as sp_stats

        x_fit = np.linspace(x.min() * 0.95, x.max() * 1.05, 100)
        y_fit = reg_stats["slope"] * x_fit + reg_stats["intercept"]

        n = len(x)
        t_val = sp_stats.t.ppf(0.975, n - 2)
        s_err = reg_stats["std_err"]

        x_mean = np.mean(x)
        sxx = np.sum((x - x_mean) ** 2)
        se_fit = s_err * np.sqrt(1 / n + (x_fit - x_mean) ** 2 / sxx)
        ci = t_val * se_fit

        ax.plot(
            x_fit, y_fit, "-", color=CB_ORANGE, linewidth=3, label="Best fit", zorder=1
        )

        ax.fill_between(
            x_fit,
            y_fit - ci,
            y_fit + ci,
            alpha=0.2,
            color=CB_ORANGE,
            label="95% CI",
            zorder=0,
        )

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
                color=CB_PINK,
                linewidth=2.5,
                alpha=0.8,
                label="Min slope",
                zorder=1,
            )
            ax.plot(
                x_fit,
                y_fit_max,
                ":",
                color=CB_PINK,
                linewidth=2.5,
                alpha=0.8,
                label="Max slope",
                zorder=1,
            )

            slope_uncertainty = abs(slope_max - slope_min) / 2
            intercept_uncertainty = abs(intercept_max - intercept_min) / 2
        else:
            slope_min = slope_max = slope_uncertainty = intercept_uncertainty = None

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
            else:
                return formatted

        textstr = f"Equation: $\\tau = m \\cdot h + c$\n"
        textstr += f"$m = {reg_stats['slope']:.6f} \\pm {format_uncertainty(slope_uncertainty)} \\, \\mathrm{{s/mm}}$\n"
        textstr += f"$c = {reg_stats['intercept']:.4f} \\pm {format_uncertainty(intercept_uncertainty)} \\, \\mathrm{{s}}$\n"
        textstr += f"$R^2 = {reg_stats['r_squared']:.4f}$\t"
        textstr += f"$p = {reg_stats['p_value']:.4f}$"
        if reg_stats["p_value"] < 0.05:
            textstr += " *"

        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"
            ),
            fontsize=14,
            family="monospace",
        )

    ax.set_xlabel("Pad Thickness ($h$, mm)", fontsize=16, fontweight="bold")
    ax.tick_params(axis="x", labelsize=16)
    ax.set_ylabel("Contact Duration ($\\tau$, s)", fontsize=16, fontweight="bold")
    ax.tick_params(axis="y", labelsize=16)
    ax.set_title(
        "Contact Duration ($\\tau$) vs. Pad Thickness ($h$)",
        fontsize=21,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.legend(loc="best", fontsize=14, framealpha=0.95, edgecolor="black")

    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_annotated_force_profiles(
    aligned_csv_path: Path,
    contact_results: pd.DataFrame,
    output_dir: Path,
    method: str = "threshold",
):
    """
    Plot force-time profiles with contact duration windows annotated.
    Uses colorblind-friendly colors and clear annotations.
    """
    df = pd.read_csv(aligned_csv_path, header=[0, 1])
    runs = detect_runs(df.columns)
    thickness = extract_thickness_from_filename(aligned_csv_path.stem)

    file_results = contact_results[
        (contact_results["file"] == aligned_csv_path.stem)
        & (contact_results["method"] == method)
    ]

    CB_COLORS = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161", "#949494"]

    fig, ax = plt.subplots(figsize=(14, 7))

    contact_windows = []

    for idx, run in enumerate(runs):
        try:
            time = df[(run, "Time (s) [aligned]")].values
            force = df[(run, "Force (N)")].values

            mask = np.isfinite(time) & np.isfinite(force)
            if not mask.any():
                continue

            color = CB_COLORS[idx % len(CB_COLORS)]
            ax.plot(
                time[mask], force[mask], label=run, linewidth=2, alpha=0.8, color=color
            )

            run_contact = file_results[file_results["run"] == run]
            if len(run_contact) > 0:
                start = run_contact.iloc[0]["start_time_s"]
                end = run_contact.iloc[0]["end_time_s"]

                if np.isfinite(start) and np.isfinite(end):
                    contact_windows.append((start, end))

        except Exception as e:
            print(f"Warning: Could not plot {run}: {e}")
            continue

    if contact_windows:
        all_starts = [w[0] for w in contact_windows]
        all_ends = [w[1] for w in contact_windows]
        min_start = min(all_starts)
        max_end = max(all_ends)

        ax.axvspan(
            min_start,
            max_end,
            alpha=0.15,
            color="#E69F00",
            label="Contact window range",
            zorder=0,
        )

        ax.axvline(
            min_start,
            color="#029E73",
            linestyle="--",
            linewidth=2.5,
            alpha=0.8,
            label="First contact start",
        )
        ax.axvline(
            max_end,
            color="#CC78BC",
            linestyle="--",
            linewidth=2.5,
            alpha=0.8,
            label="Last contact end",
        )

    ax.set_xlabel("Time (s) [aligned]", fontsize=16, fontweight="bold")
    ax.set_ylabel("Force (N)", fontsize=16, fontweight="bold")
    ax.set_title(
        f"Force Profiles with Contact Duration Windows\n{aligned_csv_path.stem} (Thickness: {thickness} mm)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.axhline(0, color="black", linewidth=1, linestyle="-", alpha=0.4)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    ax.legend(loc="best", fontsize=14, framealpha=0.95, edgecolor="black", ncol=2)
    plt.tight_layout(pad=1.5)

    output_path = output_dir / f"{aligned_csv_path.stem}_contact_annotated.png"
    plt.savefig(output_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    """
    Main analysis pipeline for contact duration analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze contact duration from aligned impact data."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="outputs",
        help="Directory containing aligned CSV files (default: outputs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="contact_analysis",
        help="Directory to save analysis results (default: contact_analysis)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Force threshold fraction for contact detection (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="threshold",
        choices=["threshold", "velocity", "both"],
        help="Contact detection method (default: threshold)",
    )
    parser.add_argument(
        "--no_smooth", action="store_true", help="Disable signal smoothing"
    )
    parser.add_argument(
        "--plots", action="store_true", help="Generate annotated force profile plots"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned_files = []
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            aligned_csv = subdir / f"{subdir.name}_aligned.csv"
            if aligned_csv.exists():
                aligned_files.append(aligned_csv)

    if not aligned_files:
        print(f"\nNo aligned CSV files found in {input_dir.resolve()}")
        print("Please run offset.py first to generate aligned data.")
        return

    all_results = []

    for aligned_csv in aligned_files:
        try:
            results = analyze_contact_duration_for_file(
                aligned_csv,
                threshold_fraction=args.threshold,
                smooth=not args.no_smooth,
                method=args.method,
            )
            all_results.append(results)
        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not all_results:
        print("\nNo results generated. Exiting.")
        return

    combined_results = pd.concat(all_results, ignore_index=True)

    detailed_path = output_dir / "contact_duration_detailed.csv"
    combined_results.to_csv(detailed_path, index=False)

    aggregated = aggregate_by_thickness(combined_results)

    if len(aggregated) == 0:
        print("\nNo valid data for aggregation.")
        return

    aggregated_path = output_dir / "contact_duration_summary.csv"
    aggregated.to_csv(aggregated_path, index=False)

    methods_to_analyze = (
        ["threshold"]
        if args.method == "threshold"
        else ["threshold", "velocity"] if args.method == "both" else ["velocity"]
    )

    for method in methods_to_analyze:
        reg_stats = perform_regression_analysis(aggregated, method)

        plot_path = output_dir / f"contact_duration_vs_thickness_{method}.png"
        plot_duration_vs_thickness(
            combined_results, aggregated, plot_path, method=method
        )

    if args.plots:
        plots_dir = output_dir / "annotated_profiles"
        plots_dir.mkdir(exist_ok=True)

        for aligned_csv in aligned_files:
            try:
                plot_annotated_force_profiles(
                    aligned_csv,
                    combined_results,
                    plots_dir,
                    method="threshold",
                )
            except Exception as e:
                print(f"Warning: Could not create plot for {aligned_csv.name}: {e}")


if __name__ == "__main__":
    main()
