"""
Analyzes aligned force data to extract physical quantities and validate
theoretical models for impact-absorbing pads.

Calculates contact duration, impulse, average force, coefficient of restitution,
and validates the impulse-momentum theorem predictions.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "serif"],
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "errorbar.capsize": 3,
    }
)


def calculate_contact_duration(
    aligned_df: pd.DataFrame, force_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Calculate contact duration τ for each run.
    Duration is measured from when |Force| exceeds threshold until it drops below again.

    Args:
        aligned_df: Aligned force data with MultiIndex columns
        force_threshold: Minimum force (N) to consider contact active

    Returns:
        DataFrame with columns: Run, Contact Duration (s), Start Time (s), End Time (s)
    """
    runs = sorted(set([c[0] for c in aligned_df.columns]))
    results = []

    for run in runs:
        t = aligned_df[(run, "Time (s) [aligned]")].values
        f = aligned_df[(run, "Force (N)")].values
        mask = np.isfinite(t) & np.isfinite(f)

        if not mask.any():
            continue

        above_threshold = np.abs(f[mask]) > force_threshold

        if not above_threshold.any():
            continue

        contact_indices = np.where(above_threshold)[0]
        start_idx = contact_indices[0]
        end_idx = contact_indices[-1]

        t_start = t[mask][start_idx]
        t_end = t[mask][end_idx]
        tau = t_end - t_start

        results.append(
            {
                "Run": run,
                "Contact Duration (s)": tau,
                "Start Time (s)": t_start,
                "End Time (s)": t_end,
            }
        )

    return pd.DataFrame(results)


def calculate_impulse_and_avg_force(
    aligned_df: pd.DataFrame, contact_durations: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate impulse J and average force F̄ for each run.

    J = ∫|F(t)|dt (area under absolute force-time curve during contact)
    F̄ = J/τ (average force from Eq. 4)

    Args:
        aligned_df: Aligned force data with MultiIndex columns
        contact_durations: DataFrame from calculate_contact_duration

    Returns:
        DataFrame with columns: Run, Impulse (N·s), Avg Force (N), Peak Force (N)

    Notes:
        - Impulse is computed using trapezoidal integration of |F(t)|
        - This assumes the force has already been zero-offset corrected
        - Peak force is the maximum absolute value during contact
    """
    results = []

    for _, row in contact_durations.iterrows():
        run = row["Run"]
        t_start = row["Start Time (s)"]
        t_end = row["End Time (s)"]
        tau = row["Contact Duration (s)"]

        t = aligned_df[(run, "Time (s) [aligned]")].values
        f = aligned_df[(run, "Force (N)")].values
        mask = np.isfinite(t) & np.isfinite(f)

        contact_mask = mask & (t >= t_start) & (t <= t_end)
        t_contact = t[contact_mask]
        f_contact = f[contact_mask]

        if len(t_contact) < 2:
            continue

        impulse = np.trapezoid(np.abs(f_contact), t_contact)
        avg_force = impulse / tau if tau > 0 else np.nan
        peak_force = np.max(np.abs(f_contact))

        results.append(
            {
                "Run": run,
                "Impulse (N·s)": impulse,
                "Avg Force (N)": avg_force,
                "Peak Force (N)": peak_force,
            }
        )

    return pd.DataFrame(results)


def calculate_coefficient_of_restitution(
    aligned_df: pd.DataFrame, contact_durations: pd.DataFrame, window_size: int = 5
) -> pd.DataFrame:
    """
    Calculate coefficient of restitution e = |v_f| / |v_i| (Eq. 5)

    v_i: velocity just before impact (at contact start)
    v_f: velocity just after impact (at contact end)

    Args:
        aligned_df: Aligned force data with MultiIndex columns
        contact_durations: DataFrame from calculate_contact_duration
        window_size: Number of points to average for velocity estimation

    Returns:
        DataFrame with columns: Run, v_initial (m/s), v_final (m/s), e

    Notes:
        - e = 1 for perfectly elastic collision (no energy loss)
        - e = 0 for perfectly inelastic collision (no rebound)
        - Velocities are averaged over window_size points to reduce noise
    """
    results = []

    for _, row in contact_durations.iterrows():
        run = row["Run"]
        t_start = row["Start Time (s)"]
        t_end = row["End Time (s)"]

        t = aligned_df[(run, "Time (s) [aligned]")].values
        v = aligned_df[(run, "Velocity (m/s)")].values
        mask = np.isfinite(t) & np.isfinite(v)

        buffer = 0.01
        before_mask = mask & (t < t_start - buffer)
        after_mask = mask & (t > t_end + buffer)

        v_i = (
            np.mean(v[before_mask][-window_size:])
            if before_mask.sum() >= window_size
            else np.nan
        )
        v_f = (
            np.mean(v[after_mask][:window_size])
            if after_mask.sum() >= window_size
            else np.nan
        )

        e = abs(v_f) / abs(v_i) if np.isfinite(v_i) and abs(v_i) > 0 else np.nan

        results.append(
            {"Run": run, "v_initial (m/s)": v_i, "v_final (m/s)": v_f, "e": e}
        )

    return pd.DataFrame(results)


def analyze_single_file(
    aligned_csv: Path, force_threshold: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform complete analysis on a single aligned CSV file.

    Args:
        aligned_csv: Path to aligned CSV file
        force_threshold: Minimum force (N) to consider contact active

    Returns:
        Tuple of (per_run_results, summary_stats)
    """
    aligned_df = pd.read_csv(aligned_csv, header=[0, 1])

    contact_dur = calculate_contact_duration(aligned_df, force_threshold)
    impulse_data = calculate_impulse_and_avg_force(aligned_df, contact_dur)
    restitution_data = calculate_coefficient_of_restitution(aligned_df, contact_dur)

    per_run = contact_dur.merge(impulse_data, on="Run").merge(
        restitution_data, on="Run"
    )

    summary = {
        "τ_mean (s)": per_run["Contact Duration (s)"].mean(),
        "τ_std (s)": per_run["Contact Duration (s)"].std(),
        "F̄_mean (N)": per_run["Avg Force (N)"].mean(),
        "F̄_std (N)": per_run["Avg Force (N)"].std(),
        "J_mean (N·s)": per_run["Impulse (N·s)"].mean(),
        "J_std (N·s)": per_run["Impulse (N·s)"].std(),
        "e_mean": per_run["e"].mean(),
        "e_std": per_run["e"].std(),
        "Peak_F_mean (N)": per_run["Peak Force (N)"].mean(),
        "Peak_F_std (N)": per_run["Peak Force (N)"].std(),
    }

    return per_run, pd.DataFrame([summary])


def aggregate_thickness_analysis(
    outputs_root: Path, thickness_map: Dict[str, float], force_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Aggregate analysis results across all pad thicknesses.

    Args:
        outputs_root: Root output directory containing aligned CSVs
        thickness_map: Dict mapping file stems to thickness in mm
                      e.g., {"dr lee go brr - 4mm": 4.0, ...}
        force_threshold: Minimum force (N) to consider contact active

    Returns:
        DataFrame with averaged quantities for each thickness
    """
    all_results = []

    for file_stem, thickness in thickness_map.items():
        file_dir = outputs_root / file_stem
        aligned_csv = file_dir / f"{file_stem}_aligned.csv"

        if not aligned_csv.exists():
            print(f"Warning: {aligned_csv} not found, skipping.")
            continue

        _, summary = analyze_single_file(aligned_csv, force_threshold)

        result_row = {"Thickness (mm)": thickness}
        result_row.update(summary.iloc[0].to_dict())
        all_results.append(result_row)

    return pd.DataFrame(all_results).sort_values("Thickness (mm)")


def plot_theory_validation(
    summary_df: pd.DataFrame, output_dir: Path, show: bool = False
) -> None:
    """
    Create publication-quality validation plots for Equations (11) and (12):
    1. F̄ vs 1/τ (should be linear through origin, slope = J)
    2. 1/F̄ vs d (should be linear, extracts τ₀ and α)

    Args:
        summary_df: DataFrame from aggregate_thickness_analysis
        output_dir: Directory to save plots
        show: Whether to display plots interactively
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    inv_tau = 1 / summary_df["τ_mean (s)"]
    inv_tau_err = summary_df["τ_std (s)"] / (summary_df["τ_mean (s)"] ** 2)
    f_bar = summary_df["F̄_mean (N)"]
    f_bar_err = summary_df["F̄_std (N)"]

    ax1.errorbar(
        inv_tau,
        f_bar,
        xerr=inv_tau_err,
        yerr=f_bar_err,
        fmt="o",
        color="#2E86AB",
        ecolor="#6C757D",
        capsize=4,
        markersize=7,
        label="Experimental data",
        zorder=3,
    )

    weights = 1 / (f_bar_err**2)
    slope = np.sum(weights * inv_tau * f_bar) / np.sum(weights * inv_tau**2)

    ss_res = np.sum((f_bar - slope * inv_tau) ** 2)
    ss_tot = np.sum((f_bar - f_bar.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    slope_err = np.sqrt(1 / np.sum(weights * inv_tau**2))

    x_fit = np.linspace(0, inv_tau.max() * 1.05, 100)
    ax1.plot(
        x_fit,
        slope * x_fit,
        "--",
        color="#A23B72",
        linewidth=2,
        label=f"Linear fit: $J = {slope:.4f} \\pm {slope_err:.4f}$ N·s\n$R^2 = {r_squared:.4f}$",
        zorder=2,
    )

    ax1.set_xlabel(r"$1/\tau$ (s$^{-1}$)")
    ax1.set_ylabel(r"$\bar{F}$ (N)")
    ax1.set_title("Eq. (11): Impulse-Momentum Validation", pad=10)
    ax1.legend(frameon=True, fancybox=False, edgecolor="black", framealpha=0.95)
    ax1.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    thickness = summary_df["Thickness (mm)"]
    inv_f_bar = 1 / f_bar
    inv_f_bar_err = f_bar_err / (f_bar**2)

    ax2.errorbar(
        thickness,
        inv_f_bar,
        yerr=inv_f_bar_err,
        fmt="s",
        color="#F18F01",
        ecolor="#6C757D",
        capsize=4,
        markersize=7,
        label="Experimental data",
        zorder=3,
    )

    coeffs = np.polyfit(thickness, inv_f_bar, 1, w=1 / inv_f_bar_err)
    fit_line = np.poly1d(coeffs)

    y_fit = fit_line(thickness)
    ss_res_2 = np.sum((inv_f_bar - y_fit) ** 2)
    ss_tot_2 = np.sum((inv_f_bar - inv_f_bar.mean()) ** 2)
    r_squared_2 = 1 - (ss_res_2 / ss_tot_2)

    cov = np.linalg.inv(
        np.array(
            [
                [np.sum(thickness**2), np.sum(thickness)],
                [np.sum(thickness), len(thickness)],
            ]
        )
        / inv_f_bar_err.mean() ** 2
    )
    slope_err_2 = np.sqrt(cov[0, 0])
    intercept_err = np.sqrt(cov[1, 1])

    x_fit2 = np.linspace(thickness.min() * 0.95, thickness.max() * 1.05, 100)
    ax2.plot(
        x_fit2,
        fit_line(x_fit2),
        "--",
        color="#C73E1D",
        linewidth=2,
        label=f"Linear fit: $R^2 = {r_squared_2:.4f}$",
        zorder=2,
    )

    J_est = slope
    J_err = slope_err
    alpha_over_J = coeffs[0]
    tau0_over_J = coeffs[1]

    tau_0 = tau0_over_J * J_est
    alpha = alpha_over_J * J_est

    tau_0_err = np.sqrt((tau0_over_J * J_err) ** 2 + (J_est * intercept_err) ** 2)
    alpha_err = np.sqrt((alpha_over_J * J_err) ** 2 + (J_est * slope_err_2) ** 2)

    ax2.set_xlabel(r"Pad thickness $d$ (mm)")
    ax2.set_ylabel(r"$1/\bar{F}$ (N$^{-1}$)")
    ax2.set_title("Eq. (12): Contact Duration Model", pad=10)
    ax2.legend(frameon=True, fancybox=False, edgecolor="black", framealpha=0.95)
    ax2.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    out_path = output_dir / "theory_validation.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Theory validation plot saved to {out_path}")

    print("\n" + "=" * 60)
    print("THEORETICAL MODEL PARAMETERS (with uncertainties)")
    print("=" * 60)
    print(f"Eq. (11) - Impulse from F̄ vs 1/τ fit:")
    print(f"  J = {J_est:.4f} ± {J_err:.4f} N·s")
    print(f"  R² = {r_squared:.4f}")
    print(f"\nEq. (12) - Contact duration model parameters:")
    print(f"  τ₀ (baseline contact time) = {tau_0:.6f} ± {tau_0_err:.6f} s")
    print(f"  α  (thickness coefficient) = {alpha:.6f} ± {alpha_err:.6f} s/mm")
    print(f"  R² = {r_squared_2:.4f}")
    print("=" * 60)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_thickness_trends(
    summary_df: pd.DataFrame, output_dir: Path, show: bool = False
) -> None:
    """
    Create publication-quality plots showing physical quantities vs pad thickness.

    Args:
        summary_df: DataFrame from aggregate_thickness_analysis
        output_dir: Directory to save plots
        show: Whether to display plots interactively
    """
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    thickness = summary_df["Thickness (mm)"]

    colors = ["#2E86AB", "#F18F01", "#C73E1D", "#06A77D"]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(
        thickness,
        summary_df["τ_mean (s)"],
        yerr=summary_df["τ_std (s)"],
        fmt="o-",
        color=colors[0],
        ecolor="#6C757D",
        capsize=4,
        markersize=7,
        linewidth=1.5,
        label="Experimental data",
    )
    ax1.set_xlabel(r"Pad thickness $d$ (mm)")
    ax1.set_ylabel(r"Contact duration $\tau$ (s)")
    ax1.set_title("(a) Contact Duration vs Thickness", loc="left", fontweight="bold")
    ax1.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax1.set_ylim(bottom=0)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.errorbar(
        thickness,
        summary_df["F̄_mean (N)"],
        yerr=summary_df["F̄_std (N)"],
        fmt="s-",
        color=colors[1],
        ecolor="#6C757D",
        capsize=4,
        markersize=7,
        linewidth=1.5,
        label="Experimental data",
    )
    ax2.set_xlabel(r"Pad thickness $d$ (mm)")
    ax2.set_ylabel(r"Average force $\bar{F}$ (N)")
    ax2.set_title("(b) Average Force vs Thickness", loc="left", fontweight="bold")
    ax2.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax2.set_ylim(bottom=0)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.errorbar(
        thickness,
        summary_df["J_mean (N·s)"],
        yerr=summary_df["J_std (N·s)"],
        fmt="^-",
        color=colors[2],
        ecolor="#6C757D",
        capsize=4,
        markersize=7,
        linewidth=1.5,
        label="Experimental data",
    )

    j_mean_all = summary_df["J_mean (N·s)"].mean()
    ax3.axhline(
        j_mean_all,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Mean: {j_mean_all:.4f} N·s",
    )

    ax3.set_xlabel(r"Pad thickness $d$ (mm)")
    ax3.set_ylabel(r"Impulse $J$ (N·s)")
    ax3.set_title(
        "(c) Impulse vs Thickness (Expected: Constant)", loc="left", fontweight="bold"
    )
    ax3.legend(frameon=True, fancybox=False, edgecolor="black", framealpha=0.95)
    ax3.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax3.set_ylim(bottom=0)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.errorbar(
        thickness,
        summary_df["e_mean"],
        yerr=summary_df["e_std"],
        fmt="d-",
        color=colors[3],
        ecolor="#6C757D",
        capsize=4,
        markersize=7,
        linewidth=1.5,
        label="Experimental data",
    )
    ax4.set_xlabel(r"Pad thickness $d$ (mm)")
    ax4.set_ylabel(r"Coefficient of restitution $e$")
    ax4.set_title("(d) Energy Dissipation vs Thickness", loc="left", fontweight="bold")
    ax4.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax4.set_ylim([0, 1.0])

    ax4.axhline(
        1.0,
        color="gray",
        linestyle=":",
        linewidth=1.0,
        alpha=0.5,
        label="Perfectly elastic",
    )
    ax4.axhline(
        0.0,
        color="gray",
        linestyle=":",
        linewidth=1.0,
        alpha=0.5,
        label="Perfectly inelastic",
    )
    ax4.legend(
        frameon=True, fancybox=False, edgecolor="black", framealpha=0.95, fontsize=9
    )

    plt.suptitle("Physical Quantities vs Pad Thickness", fontsize=14, fontweight="bold")

    out_path = output_dir / "thickness_trends.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Thickness trends plot saved to {out_path}")

    print("\n" + "=" * 60)
    print("IMPULSE CONSISTENCY CHECK")
    print("=" * 60)
    print("Theory predicts J should be constant (independent of thickness)")
    print(f"Mean J across all thicknesses: {j_mean_all:.4f} N·s")
    print(f"Std dev of J values: {summary_df['J_mean (N·s)'].std():.4f} N·s")
    print(
        f"Coefficient of variation: {summary_df['J_mean (N·s)'].std() / j_mean_all * 100:.2f}%"
    )
    print("=" * 60)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    """
    Analyze aligned force data across multiple pad thicknesses.

    --output_dir: directory containing aligned CSV outputs (default: ./outputs)
    --force_threshold: minimum force (N) to detect contact (default: 0.5)
    --show: display plots interactively in addition to saving PNGs
    """
    parser = argparse.ArgumentParser(
        description="Analyze aligned force data and validate theoretical models."
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--force_threshold", type=float, default=0.5)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Error: Output directory {output_dir.resolve()} not found.")
        print("Run offset.py first to generate aligned data.")
        return

    thickness_map = {
        "dr lee go brr - 4mm": 4.0,
        "dr lee go brr - 28mm": 28.0,
        "dr lee go brr - 40mm": 40.0,
        "dr lee go brr - 52mm": 52.0,
    }

    print("=== Analyzing individual files ===")
    for file_stem in thickness_map.keys():
        file_dir = output_dir / file_stem
        aligned_csv = file_dir / f"{file_stem}_aligned.csv"

        if not aligned_csv.exists():
            print(f"Skipping {file_stem} (aligned CSV not found)")
            continue

        print(f"\nProcessing {file_stem}...")
        per_run, summary = analyze_single_file(aligned_csv, args.force_threshold)

        per_run_path = file_dir / f"{file_stem}_analysis.csv"
        per_run.to_csv(per_run_path, index=False)
        print(f"  Per-run analysis saved to {per_run_path}")
        print(
            f"  Summary: τ = {summary['τ_mean (s)'].values[0]:.6f} s, "
            f"F̄ = {summary['F̄_mean (N)'].values[0]:.2f} N, "
            f"J = {summary['J_mean (N·s)'].values[0]:.4f} N·s, "
            f"e = {summary['e_mean'].values[0]:.3f}"
        )

    print("\n=== Aggregating thickness analysis ===")
    summary_df = aggregate_thickness_analysis(
        output_dir, thickness_map, args.force_threshold
    )

    summary_path = output_dir / "thickness_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nThickness summary saved to {summary_path}")
    print("\n" + summary_df.to_string(index=False))

    print("\n=== Generating validation plots ===")
    plot_theory_validation(summary_df, output_dir, args.show)
    plot_thickness_trends(summary_df, output_dir, args.show)

    print("\n=== Analysis complete ===")


if __name__ == "__main__":
    main()
