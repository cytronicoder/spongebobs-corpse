"""
Compare contact duration detection across multiple threshold values.

This script tests Method A (Force-Threshold) with different threshold percentages
(1%, 3%, 5%, 10% of peak force) to investigate the sensitivity of contact duration
to threshold selection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from typing import Dict, List
import argparse


def smooth_signal(
    data: np.ndarray, window_length: int = 11, polyorder: int = 3
) -> np.ndarray:
    """Apply Savitzky-Golay filter to smooth noisy signals."""
    if len(data) < window_length:
        return data

    mask = np.isfinite(data)
    if not mask.any():
        return data

    smoothed = data.copy()
    if mask.sum() >= window_length:
        smoothed[mask] = savgol_filter(data[mask], window_length, polyorder)

    return smoothed


def method_force_threshold(
    time: np.ndarray,
    force: np.ndarray,
    threshold_fraction: float = 0.05,
    smooth: bool = True,
) -> Dict[str, float]:
    """
    Force-Threshold Method with configurable threshold.

    Args:
        time: Time array (s)
        force: Force array (N)
        threshold_fraction: Threshold as fraction of peak force
        smooth: Whether to apply Savitzky-Golay smoothing

    Returns:
        Dictionary with contact metrics
    """
    mask = np.isfinite(time) & np.isfinite(force)
    if not mask.any():
        return {
            "duration": np.nan,
            "start_time": np.nan,
            "end_time": np.nan,
            "threshold_fraction": threshold_fraction,
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
        "threshold": threshold,
        "threshold_fraction": threshold_fraction,
    }


def load_single_run(csv_path: Path, run: str = "Run 1") -> tuple:
    """Load time and force data for a single run."""
    df = pd.read_csv(csv_path, header=[0, 1])
    time = df[(run, "Time (s) [aligned]")].values
    force = df[(run, "Force (N)")].values
    return time, force


def compare_thresholds(
    time: np.ndarray,
    force: np.ndarray,
    threshold_fractions: List[float] = [0.01, 0.03, 0.05, 0.10],
) -> pd.DataFrame:
    """
    Compare contact duration across multiple threshold values.

    Args:
        time: Time array (s)
        force: Force array (N)
        threshold_fractions: List of threshold fractions to test

    Returns:
        DataFrame with results for each threshold
    """
    results = []
    for frac in threshold_fractions:
        result = method_force_threshold(time, force, frac)
        results.append(result)

    return pd.DataFrame(results)


def plot_threshold_comparison(
    time: np.ndarray,
    force: np.ndarray,
    results_df: pd.DataFrame,
    save_path: Path = None,
):
    """
    Create visualization comparing all threshold values.

    Args:
        time: Time array (s)
        force: Force array (N)
        results_df: DataFrame with threshold comparison results
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.plot(time, force, "b-", linewidth=2, label="Force signal", alpha=0.7, zorder=1)

    colors = ["red", "orange", "green", "purple"]
    linestyles = ["-", "--", "-.", ":"]

    for idx, (_, row) in enumerate(results_df.iterrows()):
        if np.isnan(row["duration"]):
            continue

        threshold_pct = row["threshold_fraction"] * 100
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]

        ax.axhline(
            y=row["threshold"],
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f"{threshold_pct:.0f}% threshold ({row['threshold']:.3f} N)",
            zorder=2,
        )
        ax.axhline(
            y=-row["threshold"],
            color=color,
            linestyle=linestyle,
            linewidth=2,
            zorder=2,
        )

        ax.axvspan(
            row["start_time"],
            row["end_time"],
            alpha=0.15,
            color=color,
            zorder=0,
        )

        ax.axvline(
            x=row["start_time"],
            color=color,
            linestyle="-",
            linewidth=1.5,
            alpha=0.6,
            zorder=2,
        )
        ax.axvline(
            x=row["end_time"],
            color=color,
            linestyle="-",
            linewidth=1.5,
            alpha=0.6,
            zorder=2,
        )

        mid_time = (row["start_time"] + row["end_time"]) / 2
        y_positions = [0.8, 0.6, 0.4, 0.2]
        y_pos = row["peak_force"] * y_positions[idx % len(y_positions)]

        ax.annotate(
            f"{threshold_pct:.0f}%: τ = {row['duration']*1000:.2f} ms",
            xy=(mid_time, y_pos),
            fontsize=11,
            ha="center",
            bbox=dict(boxstyle="round", facecolor=color, alpha=0.3, edgecolor=color),
            zorder=3,
        )

    ax.set_xlabel("Time (s)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Force (N)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Contact Duration Sensitivity to Force Threshold",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to {save_path}")

    plt.show()


def create_summary_table(results_df: pd.DataFrame) -> None:
    """
    Print summary table of threshold comparison.

    Args:
        results_df: DataFrame with threshold comparison results
    """
    print("\n" + "=" * 80)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"{'Threshold':<15} {'Duration (ms)':<15} {'Start (s)':<12} {'End (s)':<12}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        threshold_pct = row["threshold_fraction"] * 100
        duration_ms = (
            row["duration"] * 1000 if not np.isnan(row["duration"]) else np.nan
        )

        print(
            f"{threshold_pct:>6.0f}% ({row['threshold']:.3f} N)  "
            f"{duration_ms:>10.2f}      "
            f"{row['start_time']:>8.4f}    "
            f"{row['end_time']:>8.4f}"
        )

    durations = results_df["duration"].values * 1000
    valid_durations = durations[~np.isnan(durations)]

    if len(valid_durations) > 1:
        mean_duration = np.mean(valid_durations)
        std_duration = np.std(valid_durations)
        cv = (std_duration / mean_duration) * 100
        duration_range = np.max(valid_durations) - np.min(valid_durations)

        print("-" * 80)
        print(f"Mean Duration: {mean_duration:.2f} ms")
        print(f"Std Deviation: {std_duration:.2f} ms")
        print(f"Coefficient of Variation: {cv:.2f}%")
        print(f"Duration Range: {duration_range:.2f} ms")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare contact duration across multiple force threshold values"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to aligned CSV file",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="Run 1",
        help="Run identifier (e.g., 'Run 1', 'Run 2')",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.05, 0.10],
        help="Threshold fractions to test (default: 0.01 0.03 0.05 0.10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="threshold_comparison.png",
        help="Output filename for the plot",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="threshold_comparison.csv",
        help="Output filename for CSV results",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    print(f"Loading data from: {csv_path}")
    print(f"Analyzing run: {args.run}")
    print(f"Testing thresholds: {[f'{t*100:.0f}%' for t in args.thresholds]}\n")

    time, force = load_single_run(csv_path, args.run)

    results_df = compare_thresholds(time, force, args.thresholds)

    create_summary_table(results_df)

    results_df.to_csv(args.csv_output, index=False)
    print(f"Results saved to: {args.csv_output}")

    plot_threshold_comparison(time, force, results_df, Path(args.output))


if __name__ == "__main__":
    main()
