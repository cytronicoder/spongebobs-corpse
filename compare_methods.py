"""
Compare contact duration detection methods for preliminary investigation.

This script compares three methods:
1. Force-threshold detection
2. Velocity-based zero-crossing analysis
3. Energy-based kinetic energy tracking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from typing import Dict, Tuple
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
    Force-Threshold Method (Method A).

    Identifies contact using force-sensor data. Contact is defined as the interval
    during which |F(t)| exceeds a threshold F_threshold = α * F_peak.

    Args:
        time: Time array (s)
        force: Force array (N)
        threshold_fraction: Threshold as fraction of peak force (α)
        smooth: Whether to apply Savitzky-Golay smoothing

    Returns:
        Dictionary with contact metrics including duration, start, end times
    """
    mask = np.isfinite(time) & np.isfinite(force)
    if not mask.any():
        return {
            "duration": np.nan,
            "start_time": np.nan,
            "end_time": np.nan,
            "method": "Force-Threshold",
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
        "method": "Force-Threshold",
    }


def method_velocity_based(
    time: np.ndarray,
    velocity: np.ndarray,
    smooth: bool = True,
    velocity_threshold: float = 0.1,
) -> Dict[str, float]:
    """
    Velocity-Based Method (Method B).

    Defines contact using velocity-time data. Contact begins when velocity drops below
    a threshold and ends when it recovers above the threshold after minimum velocity.
    This handles sparse velocity data with NaN values.

    Args:
        time: Time array (s)
        velocity: Velocity array (m/s)
        smooth: Whether to apply smoothing
        velocity_threshold: Velocity threshold for contact detection (m/s)

    Returns:
        Dictionary with contact metrics
    """
    mask = np.isfinite(time) & np.isfinite(velocity)
    if not mask.any() or mask.sum() < 10:
        return {
            "duration": np.nan,
            "start_time": np.nan,
            "end_time": np.nan,
            "method": "Velocity-Based",
        }

    t_valid = time[mask]
    v_valid = velocity[mask]

    if smooth and len(v_valid) >= 11:
        v_valid = smooth_signal(v_valid)

    min_vel_idx = np.argmin(v_valid)
    min_velocity = v_valid[min_vel_idx]

    start_idx = 0
    for i in range(min_vel_idx, -1, -1):
        if v_valid[i] > velocity_threshold:
            start_idx = i
            break

    end_idx = len(v_valid) - 1
    for i in range(min_vel_idx, len(v_valid)):
        if abs(v_valid[i]) > velocity_threshold:
            end_idx = i
            break

    start_time = t_valid[start_idx]
    end_time = t_valid[end_idx]
    duration = end_time - start_time

    return {
        "duration": duration,
        "start_time": start_time,
        "end_time": end_time,
        "min_velocity": min_velocity,
        "velocity_threshold": velocity_threshold,
        "method": "Velocity-Based",
    }


def method_energy_based(
    time: np.ndarray,
    velocity: np.ndarray,
    mass: float = 0.5,
    recovery_fraction: float = 0.5,
    smooth: bool = True,
) -> Dict[str, float]:
    """
    Energy-Based Method (Method C).

    Uses kinetic energy E_k(t) = 0.5 * m * v²(t). Contact duration is the interval
    between the initial energy drop and recovery to a specified fraction of initial energy.
    This method now uses the global maximum energy to properly identify the impact event.

    Args:
        time: Time array (s)
        velocity: Velocity array (m/s)
        mass: Cart mass (kg)
        recovery_fraction: Fraction of initial energy for contact end
        smooth: Whether to apply smoothing

    Returns:
        Dictionary with contact metrics
    """
    mask = np.isfinite(time) & np.isfinite(velocity)
    if not mask.any() or mask.sum() < 10:
        return {
            "duration": np.nan,
            "start_time": np.nan,
            "end_time": np.nan,
            "method": "Energy-Based",
        }

    t_valid = time[mask]
    v_valid = velocity[mask]

    if smooth and len(v_valid) >= 11:
        v_valid = smooth_signal(v_valid)

    energy = 0.5 * mass * v_valid**2

    max_energy_idx = np.argmax(energy)
    initial_energy = energy[max_energy_idx]

    min_energy_idx = max_energy_idx + np.argmin(energy[max_energy_idx:])
    min_energy = energy[min_energy_idx]

    recovery_threshold = recovery_fraction * initial_energy

    start_idx = max_energy_idx
    for i in range(min_energy_idx, max_energy_idx, -1):
        if energy[i] >= recovery_threshold:
            start_idx = i
            break

    end_idx = len(energy) - 1
    for i in range(min_energy_idx, len(energy)):
        if energy[i] >= recovery_threshold:
            end_idx = i
            break

    start_time = t_valid[start_idx]
    end_time = t_valid[end_idx]
    duration = end_time - start_time

    return {
        "duration": duration,
        "start_time": start_time,
        "end_time": end_time,
        "initial_energy": initial_energy,
        "min_energy": min_energy,
        "recovery_threshold": recovery_threshold,
        "method": "Energy-Based",
    }


def load_single_run(
    csv_path: Path, run: str = "Run 1"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load time, force, and velocity data for a single run.

    Args:
        csv_path: Path to aligned CSV file
        run: Run identifier (e.g., "Run 1")

    Returns:
        Tuple of (time, force, velocity) arrays
    """
    df = pd.read_csv(csv_path, header=[0, 1])

    time = df[(run, "Time (s) [aligned]")].values
    force = df[(run, "Force (N)")].values
    velocity = df[(run, "Velocity (m/s)")].values

    return time, force, velocity


def compare_methods_single_run(
    time: np.ndarray,
    force: np.ndarray,
    velocity: np.ndarray,
    threshold_fraction: float = 0.05,
    mass: float = 0.5,
) -> pd.DataFrame:
    """
    Compare all three methods on a single run.

    Args:
        time: Time array (s)
        force: Force array (N)
        velocity: Velocity array (m/s)
        threshold_fraction: Threshold for force method
        mass: Cart mass for energy method (kg)

    Returns:
        DataFrame with results from all three methods
    """
    results = []

    result_force = method_force_threshold(time, force, threshold_fraction)
    results.append(result_force)

    result_velocity = method_velocity_based(time, velocity)
    results.append(result_velocity)

    result_energy = method_energy_based(time, velocity, mass)
    results.append(result_energy)

    return pd.DataFrame(results)


def plot_method_comparison(
    time: np.ndarray,
    force: np.ndarray,
    velocity: np.ndarray,
    results_df: pd.DataFrame,
    threshold_fraction: float = 0.05,
    mass: float = 0.5,
    save_path: Path = None,
):
    """
    Create comprehensive visualization comparing all three methods.

    Args:
        time: Time array (s)
        force: Force array (N)
        velocity: Velocity array (m/s)
        results_df: DataFrame with method comparison results
        threshold_fraction: Threshold fraction for force method
        mass: Cart mass (kg)
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    mask = np.isfinite(velocity)
    energy = np.zeros_like(velocity)
    energy[mask] = 0.5 * mass * velocity[mask] ** 2

    force_result = results_df[results_df["method"] == "Force-Threshold"].iloc[0]
    velocity_result = results_df[results_df["method"] == "Velocity-Based"].iloc[0]
    energy_result = results_df[results_df["method"] == "Energy-Based"].iloc[0]

    ax1 = axes[0]
    ax1.plot(time, force, "b-", linewidth=1.5, label="Force signal", alpha=0.7)

    if not np.isnan(force_result["duration"]):
        ax1.axhline(
            y=force_result["threshold"],
            color="r",
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({threshold_fraction*100:.0f}% of peak)",
        )
        ax1.axhline(
            y=-force_result["threshold"], color="r", linestyle="--", linewidth=1.5
        )

        ax1.axvspan(
            force_result["start_time"],
            force_result["end_time"],
            alpha=0.2,
            color="green",
            label="Contact region",
        )

        ax1.axvline(
            x=force_result["start_time"],
            color="green",
            linestyle="-",
            linewidth=2,
            label="Contact start",
        )
        ax1.axvline(
            x=force_result["end_time"],
            color="orange",
            linestyle="-",
            linewidth=2,
            label="Contact end",
        )

        mid_time = (force_result["start_time"] + force_result["end_time"]) / 2
        ax1.annotate(
            f"τ = {force_result['duration']*1000:.2f} ms",
            xy=(mid_time, force_result["peak_force"] * 0.5),
            fontsize=12,
            ha="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Force (N)", fontsize=12)
    ax1.set_title("Method A: Force-Threshold Detection", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    v_mask = np.isfinite(velocity)
    ax2.plot(
        time[v_mask],
        velocity[v_mask],
        "b-",
        linewidth=1.5,
        label="Velocity signal",
        alpha=0.7,
        marker="o",
        markersize=3,
    )
    ax2.axhline(y=0, color="r", linestyle="--", linewidth=1.5, label="Zero velocity")

    if (
        not np.isnan(velocity_result["duration"])
        and "velocity_threshold" in velocity_result
    ):
        ax2.axhline(
            y=velocity_result["velocity_threshold"],
            color="purple",
            linestyle=":",
            linewidth=1.5,
            label=f"Threshold ({velocity_result['velocity_threshold']:.2f} m/s)",
        )
        ax2.axhline(
            y=-velocity_result["velocity_threshold"],
            color="purple",
            linestyle=":",
            linewidth=1.5,
        )

    if not np.isnan(velocity_result["duration"]):
        ax2.axvspan(
            velocity_result["start_time"],
            velocity_result["end_time"],
            alpha=0.2,
            color="green",
            label="Contact region",
        )

        ax2.axvline(
            x=velocity_result["start_time"],
            color="green",
            linestyle="-",
            linewidth=2,
            label="Contact start",
        )
        ax2.axvline(
            x=velocity_result["end_time"],
            color="orange",
            linestyle="-",
            linewidth=2,
            label="Contact end",
        )

        mid_time = (velocity_result["start_time"] + velocity_result["end_time"]) / 2
        v_range = np.nanmax(velocity[v_mask]) - np.nanmin(velocity[v_mask])
        y_pos = np.nanmin(velocity[v_mask]) + v_range * 0.3
        ax2.annotate(
            f"τ = {velocity_result['duration']*1000:.2f} ms",
            xy=(mid_time, y_pos),
            fontsize=12,
            ha="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Velocity (m/s)", fontsize=12)
    ax2.set_title(
        "Method B: Velocity-Based Threshold Detection",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(time, energy, "b-", linewidth=1.5, label="Kinetic energy", alpha=0.7)

    if not np.isnan(energy_result["duration"]):
        ax3.axhline(
            y=energy_result["recovery_threshold"],
            color="r",
            linestyle="--",
            linewidth=1.5,
            label="Recovery threshold (50% of initial)",
        )

        ax3.axvspan(
            energy_result["start_time"],
            energy_result["end_time"],
            alpha=0.2,
            color="green",
            label="Contact region",
        )

        ax3.axvline(
            x=energy_result["start_time"],
            color="green",
            linestyle="-",
            linewidth=2,
            label="Energy drop",
        )
        ax3.axvline(
            x=energy_result["end_time"],
            color="orange",
            linestyle="-",
            linewidth=2,
            label="Energy recovery",
        )

        mid_time = (energy_result["start_time"] + energy_result["end_time"]) / 2
        ax3.annotate(
            f"τ = {energy_result['duration']*1000:.2f} ms",
            xy=(mid_time, energy_result["recovery_threshold"] * 0.7),
            fontsize=12,
            ha="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    ax3.set_xlabel("Time (s)", fontsize=12)
    ax3.set_ylabel("Kinetic Energy (J)", fontsize=12)
    ax3.set_title(
        "Method C: Energy-Based Kinetic Energy Tracking", fontsize=14, fontweight="bold"
    )
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def create_summary_comparison(results_df: pd.DataFrame) -> None:
    """
    Print summary comparison of all three methods.

    Args:
        results_df: DataFrame with method comparison results
    """
    print("\n" + "=" * 70)
    print("CONTACT DURATION METHOD COMPARISON")
    print("=" * 70)

    for idx, row in results_df.iterrows():
        method = row["method"]
        duration_ms = (
            row["duration"] * 1000 if not np.isnan(row["duration"]) else np.nan
        )

        print(f"\n{method}:")
        print(f"  Contact Duration: {duration_ms:.3f} ms")
        print(f"  Start Time: {row['start_time']:.4f} s")
        print(f"  End Time: {row['end_time']:.4f} s")

        if method == "Force-Threshold" and "peak_force" in row:
            print(f"  Peak Force: {row['peak_force']:.2f} N")
            print(f"  Threshold: {row['threshold']:.2f} N")
        elif method == "Velocity-Based" and "min_velocity" in row:
            print(f"  Min Velocity: {row['min_velocity']:.4f} m/s")
        elif method == "Energy-Based" and "initial_energy" in row:
            print(f"  Initial Energy: {row['initial_energy']:.4f} J")
            print(f"  Min Energy: {row['min_energy']:.4f} J")

    if len(results_df) == 3:
        durations = results_df["duration"].values * 1000  # Convert to ms
        max_diff = np.max(durations) - np.min(durations)
        mean_duration = np.mean(durations)
        cv = (np.std(durations) / mean_duration) * 100

        print("\n" + "-" * 70)
        print("COMPARISON STATISTICS:")
        print(f"  Mean Duration: {mean_duration:.3f} ms")
        print(f"  Standard Deviation: {np.std(durations):.3f} ms")
        print(f"  Coefficient of Variation: {cv:.2f}%")
        print(f"  Maximum Difference: {max_diff:.3f} ms")
        print("=" * 70 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Compare contact duration detection methods"
    )
    parser.add_argument("aligned_csv", type=str, help="Path to aligned CSV file")
    parser.add_argument(
        "--run", type=str, default="Run 1", help="Run identifier (default: Run 1)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Force threshold fraction (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--mass", type=float, default=0.5, help="Cart mass in kg (default: 0.5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="method_comparison.png",
        help="Output filename for plot (default: method_comparison.png)",
    )

    args = parser.parse_args()

    csv_path = Path(args.aligned_csv)
    print(f"Loading data from: {csv_path}")
    print(f"Analyzing run: {args.run}")

    time, force, velocity = load_single_run(csv_path, args.run)

    results_df = compare_methods_single_run(
        time, force, velocity, threshold_fraction=args.threshold, mass=args.mass
    )

    create_summary_comparison(results_df)

    output_path = Path(args.output)
    plot_method_comparison(
        time,
        force,
        velocity,
        results_df,
        threshold_fraction=args.threshold,
        mass=args.mass,
        save_path=output_path,
    )

    csv_output = output_path.with_suffix(".csv")
    results_df.to_csv(csv_output, index=False)
    print(f"Results saved to: {csv_output}")


if __name__ == "__main__":
    main()
