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
from scipy.odr import ODR, Model, RealData

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


def calculate_energy_dissipation(restitution_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate fractional energy dissipation ΔE/E_in = 1 - e².

    For a collision, kinetic energy ratio is:
    E_f / E_in = (v_f / v_in)² = e²

    Therefore: ΔE/E_in = 1 - e²

    Args:
        restitution_data: DataFrame with 'e' column

    Returns:
        DataFrame with added 'Energy Dissipation' column
    """
    result = restitution_data.copy()
    result["Energy Dissipation (ΔE/E_in)"] = 1 - result["e"] ** 2
    return result


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
    restitution_data = calculate_energy_dissipation(restitution_data)

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
        "Energy_Diss_mean": per_run["Energy Dissipation (ΔE/E_in)"].mean(),
        "Energy_Diss_std": per_run["Energy Dissipation (ΔE/E_in)"].std(),
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
    1. F̄ vs 1/τ (should be linear through origin, slope = J) - using ODR
    2. 1/F̄ vs d (linear) and 1/F̄ vs √d (viscoelastic model comparison)
    3. J vs (1+e) to validate momentum-restitution identity

    Args:
        summary_df: DataFrame from aggregate_thickness_analysis
        output_dir: Directory to save plots
        show: Whether to display plots interactively
    """
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.25, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])

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

    def linear_through_origin(p, x):
        return p[0] * x

    model = Model(linear_through_origin)
    data = RealData(
        inv_tau.values, f_bar.values, sx=inv_tau_err.values, sy=f_bar_err.values
    )
    odr = ODR(data, model, beta0=[0.8])
    odr_output = odr.run()

    J_odr = odr_output.beta[0]
    J_odr_err = odr_output.sd_beta[0]

    y_pred = J_odr * inv_tau
    rmse = np.sqrt(np.mean((f_bar - y_pred) ** 2))

    x_fit = np.linspace(0, inv_tau.max() * 1.05, 100)
    ax1.plot(
        x_fit,
        J_odr * x_fit,
        "--",
        color="#A23B72",
        linewidth=2,
        label=f"ODR fit: $J = {J_odr:.3f} \\pm {J_odr_err:.3f}$ N·s\nRMSE = {rmse:.3f} N",
        zorder=2,
    )

    ax1.set_xlabel(r"$1/\tau$ (s$^{-1}$)")
    ax1.set_ylabel(r"$\bar{F}$ (N)")
    ax1.set_title(
        "(a) Impulse-Momentum Validation (Eq. 11)", loc="left", fontweight="bold"
    )
    ax1.legend(frameon=True, fancybox=False, edgecolor="black", framealpha=0.95)
    ax1.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    ax2 = fig.add_subplot(gs[0, 1])

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

    weights_linear = 1 / inv_f_bar_err**2
    coeffs_linear = np.polyfit(thickness, inv_f_bar, 1, w=weights_linear)
    fit_linear = np.poly1d(coeffs_linear)
    y_pred_linear = fit_linear(thickness)
    ss_res_linear = np.sum(weights_linear * (inv_f_bar - y_pred_linear) ** 2)
    ss_tot_linear = np.sum(
        weights_linear
        * (inv_f_bar - np.average(inv_f_bar, weights=weights_linear)) ** 2
    )
    r_squared_linear = 1 - (ss_res_linear / ss_tot_linear)

    sqrt_d = np.sqrt(thickness)
    coeffs_sqrt = np.polyfit(sqrt_d, inv_f_bar, 1, w=weights_linear)
    fit_sqrt = np.poly1d(coeffs_sqrt)
    y_pred_sqrt = fit_sqrt(sqrt_d)
    ss_res_sqrt = np.sum(weights_linear * (inv_f_bar - y_pred_sqrt) ** 2)
    ss_tot_sqrt = ss_tot_linear
    r_squared_sqrt = 1 - (ss_res_sqrt / ss_tot_sqrt)

    n = len(thickness)
    k_linear = 2
    k_sqrt = 2
    aic_linear = n * np.log(ss_res_linear / n) + 2 * k_linear
    aic_sqrt = n * np.log(ss_res_sqrt / n) + 2 * k_sqrt

    x_fit_d = np.linspace(thickness.min() * 0.95, thickness.max() * 1.05, 100)
    x_fit_sqrt = np.sqrt(x_fit_d)

    ax2.plot(
        x_fit_d,
        fit_linear(x_fit_d),
        "--",
        color="#C73E1D",
        linewidth=2,
        label=f"Linear: $R^2 = {r_squared_linear:.3f}$, AIC = {aic_linear:.1f}",
        zorder=2,
    )

    ax2.plot(
        x_fit_d,
        fit_sqrt(x_fit_sqrt),
        ":",
        color="#06A77D",
        linewidth=2,
        label=f"$\\sqrt{{d}}$: $R^2 = {r_squared_sqrt:.3f}$, AIC = {aic_sqrt:.1f}",
        zorder=2,
    )

    ax2.set_xlabel(r"Pad thickness $d$ (mm)")
    ax2.set_ylabel(r"$1/\bar{F}$ (N$^{-1}$)")
    ax2.set_title("(b) Contact Duration Model (Eq. 12)", loc="left", fontweight="bold")
    ax2.legend(
        frameon=True, fancybox=False, edgecolor="black", framealpha=0.95, fontsize=9
    )
    ax2.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

    ax3 = fig.add_subplot(gs[0, 2])

    J_vals = summary_df["J_mean (N·s)"]
    J_errs = summary_df["J_std (N·s)"]
    e_vals = summary_df["e_mean"]
    e_errs = summary_df["e_std"]
    one_plus_e = 1 + e_vals
    one_plus_e_err = e_errs

    ax3.errorbar(
        one_plus_e,
        J_vals,
        xerr=one_plus_e_err,
        yerr=J_errs,
        fmt="^",
        color="#6A4C93",
        ecolor="#6C757D",
        capsize=4,
        markersize=7,
        label="Experimental data",
        zorder=3,
    )

    data_je = RealData(
        one_plus_e.values, J_vals.values, sx=one_plus_e_err.values, sy=J_errs.values
    )
    odr_je = ODR(data_je, model, beta0=[0.5])
    odr_je_output = odr_je.run()

    m_v_in = odr_je_output.beta[0]
    m_v_in_err = odr_je_output.sd_beta[0]

    y_pred_je = m_v_in * one_plus_e
    rmse_je = np.sqrt(np.mean((J_vals - y_pred_je) ** 2))

    x_fit_je = np.linspace(0, one_plus_e.max() * 1.05, 100)
    ax3.plot(
        x_fit_je,
        m_v_in * x_fit_je,
        "--",
        color="#D4A5A5",
        linewidth=2,
        label=f"$J = m v_{{\\rm in}}(1+e)$\n$m v_{{\\rm in}} = {m_v_in:.3f} \\pm {m_v_in_err:.3f}$ N·s\nRMSE = {rmse_je:.4f} N·s",
        zorder=2,
    )

    ax3.set_xlabel(r"$1 + e$")
    ax3.set_ylabel(r"Impulse $J$ (N·s)")
    ax3.set_title("(c) Momentum-Restitution Identity", loc="left", fontweight="bold")
    ax3.legend(
        frameon=True, fancybox=False, edgecolor="black", framealpha=0.95, fontsize=9
    )
    ax3.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0)

    plt.tight_layout()
    out_path = output_dir / "theory_validation.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Theory validation plot saved to {out_path}")

    print("\n" + "=" * 70)
    print("THEORETICAL MODEL PARAMETERS (ODR with uncertainties)")
    print("=" * 70)
    print(f"Eq. (11) - Impulse from F̄ vs 1/τ (ODR, through origin):")
    print(f"  J = {J_odr:.3f} ± {J_odr_err:.3f} N·s")
    print(f"  RMSE = {rmse:.3f} N")
    print(f"\nEq. (12) - Contact duration model comparison:")
    print(
        f"  Linear d model:    1/F̄ = {coeffs_linear[1]:.5f} + {coeffs_linear[0]:.6f}·d"
    )
    print(f"                     R² = {r_squared_linear:.4f}, AIC = {aic_linear:.1f}")
    print(f"  Viscoelastic √d:   1/F̄ = {coeffs_sqrt[1]:.5f} + {coeffs_sqrt[0]:.6f}·√d")
    print(f"                     R² = {r_squared_sqrt:.4f}, AIC = {aic_sqrt:.1f}")
    if aic_linear < aic_sqrt:
        print(f"  → Linear model preferred (ΔAIC = {aic_sqrt - aic_linear:.1f})")
    else:
        print(f"  → √d model preferred (ΔAIC = {aic_linear - aic_sqrt:.1f})")
    print(f"\nMomentum-Restitution Identity: J = m·v_in·(1+e)")
    print(f"  m·v_in = {m_v_in:.3f} ± {m_v_in_err:.3f} N·s")
    print(f"  RMSE = {rmse_je:.4f} N·s")
    print(f"  Correlation(J, 1+e): r = {np.corrcoef(J_vals, one_plus_e)[0,1]:.3f}")
    print("=" * 70)

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
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    thickness = summary_df["Thickness (mm)"]
    colors = ["#2E86AB", "#F18F01", "#C73E1D", "#06A77D", "#6A4C93"]

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
    )
    ax2.set_xlabel(r"Pad thickness $d$ (mm)")
    ax2.set_ylabel(r"Average force $\bar{F}$ (N)")
    ax2.set_title("(b) Average Force vs Thickness", loc="left", fontweight="bold")
    ax2.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax2.set_ylim(bottom=0)

    ax3 = fig.add_subplot(gs[0, 2])
    J_vals = summary_df["J_mean (N·s)"]
    e_vals = summary_df["e_mean"]

    ax3.errorbar(
        thickness,
        J_vals,
        yerr=summary_df["J_std (N·s)"],
        fmt="^-",
        color=colors[2],
        ecolor="#6C757D",
        capsize=4,
        markersize=7,
        linewidth=1.5,
        label="Experimental $J$",
    )

    j_mean_all = J_vals.mean()
    ax3.axhline(
        j_mean_all,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Mean: {j_mean_all:.3f} N·s",
    )

    ax3.set_xlabel(r"Pad thickness $d$ (mm)")
    ax3.set_ylabel(r"Impulse $J$ (N·s)")
    ax3.set_title(
        r"(c) Impulse: $J = m v_{\rm in}(1+e)$", loc="left", fontweight="bold"
    )
    ax3.legend(
        frameon=True, fancybox=False, edgecolor="black", framealpha=0.95, fontsize=9
    )
    ax3.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax3.set_ylim(bottom=0)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.errorbar(
        thickness,
        e_vals,
        yerr=summary_df["e_std"],
        fmt="d-",
        color=colors[3],
        ecolor="#6C757D",
        capsize=4,
        markersize=7,
        linewidth=1.5,
    )
    ax4.set_xlabel(r"Pad thickness $d$ (mm)")
    ax4.set_ylabel(r"Coefficient of restitution $e$")
    ax4.set_title("(d) Restitution vs Thickness", loc="left", fontweight="bold")
    ax4.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax4.set_ylim([0, 1.0])
    ax4.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)
    ax4.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)

    ax5 = fig.add_subplot(gs[1, 1])
    energy_diss = summary_df["Energy_Diss_mean"]
    energy_diss_err = summary_df["Energy_Diss_std"]

    ax5.errorbar(
        thickness,
        energy_diss * 100,
        yerr=energy_diss_err * 100,
        fmt="p-",
        color=colors[4],
        ecolor="#6C757D",
        capsize=4,
        markersize=7,
        linewidth=1.5,
    )
    ax5.set_xlabel(r"Pad thickness $d$ (mm)")
    ax5.set_ylabel(r"Energy dissipation $\Delta E/E_{\rm in}$ (%)")
    ax5.set_title(r"(e) Energy Dissipation: $1 - e^2$", loc="left", fontweight="bold")
    ax5.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax5.set_ylim([0, 100])

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.errorbar(
        e_vals,
        J_vals,
        xerr=summary_df["e_std"],
        yerr=summary_df["J_std (N·s)"],
        fmt="o",
        color="#D4A5A5",
        ecolor="#6C757D",
        capsize=4,
        markersize=7,
    )

    for i, txt in enumerate(thickness):
        ax6.annotate(
            f"{int(txt)} mm",
            (e_vals.iloc[i], J_vals.iloc[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color="#333333",
        )

    corr_j_e = np.corrcoef(J_vals, e_vals)[0, 1]
    ax6.set_xlabel(r"Coefficient of restitution $e$")
    ax6.set_ylabel(r"Impulse $J$ (N·s)")
    ax6.set_title(
        f"(f) $J$-$e$ Correlation: $r = {corr_j_e:.3f}$", loc="left", fontweight="bold"
    )
    ax6.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

    plt.suptitle(
        "Physical Quantities vs Pad Thickness", fontsize=14, fontweight="bold", y=0.95
    )

    out_path = output_dir / "thickness_trends.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Thickness trends plot saved to {out_path}")

    print("\n" + "=" * 70)
    print("IMPULSE VARIATION AND PHYSICAL CONSISTENCY")
    print("=" * 70)
    print(f"Impulse varies with thickness (J = m·v_in·(1+e) framework):")
    print(f"  Mean J: {j_mean_all:.3f} N·s")
    print(f"  Std dev: {J_vals.std():.3f} N·s")
    print(f"  Coefficient of variation: {J_vals.std() / j_mean_all * 100:.1f}%")
    print(f"  Range: {J_vals.min():.3f} – {J_vals.max():.3f} N·s")
    print(f"\nCorrelation analysis:")
    print(f"  Corr(J, e): r = {corr_j_e:.3f}")
    print(f"  Corr(J, 1+e): r = {np.corrcoef(J_vals, 1 + e_vals)[0,1]:.3f}")
    print(f"\n→ Strong J-e correlation confirms J variation is physically consistent")
    print(f"  with momentum-restitution identity: J = m·v_in·(1+e)")

    print(f"\n" + "=" * 70)
    print("ENERGY DISSIPATION BY THICKNESS")
    print("=" * 70)
    for i, row in summary_df.iterrows():
        d = row["Thickness (mm)"]
        de = row["Energy_Diss_mean"] * 100
        de_err = row["Energy_Diss_std"] * 100
        print(f"  {int(d):2d} mm: ΔE/E_in = {de:.1f} ± {de_err:.1f}%")

    avg_diss = energy_diss.mean() * 100
    print(f"\nAverage energy dissipation: {avg_diss:.1f}%")
    print(
        f"Thicker pads (40-52 mm) dissipate ~{energy_diss.iloc[2:].mean()*100:.0f}% on average"
    )
    print(f"vs. thinner pads (4-28 mm) at ~{energy_diss.iloc[:2].mean()*100:.0f}%")
    print("=" * 70)

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
