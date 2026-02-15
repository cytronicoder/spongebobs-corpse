from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_by_thickness(
    results_df: pd.DataFrame,
    time_instrument_unc: float = 0.01,
    force_instrument_unc: float = 0.05,
) -> pd.DataFrame:
    valid_df = results_df[results_df["duration_s"].notna()].copy()
    if valid_df.empty:
        return pd.DataFrame()

    grouped = (
        valid_df.groupby(["thickness_mm", "method"], as_index=False)
        .agg(
            duration_s_mean=("duration_s", "mean"),
            duration_s_std=("duration_s", "std"),
            duration_s_min=("duration_s", "min"),
            duration_s_max=("duration_s", "max"),
            duration_s_count=("duration_s", "count"),
            peak_force_N_mean=("peak_force_N", "mean"),
            peak_force_N_std=("peak_force_N", "std"),
            peak_force_N_min=("peak_force_N", "min"),
            peak_force_N_max=("peak_force_N", "max"),
        )
        .sort_values("thickness_mm")
    )

    grouped["duration_cv"] = grouped["duration_s_std"] / grouped["duration_s_mean"]

    u_random_t = grouped["duration_s_std"] / np.sqrt(grouped["duration_s_count"])
    grouped["duration_s_uncertainty"] = np.sqrt(u_random_t**2 + time_instrument_unc**2)

    u_random_f = grouped["peak_force_N_std"] / np.sqrt(grouped["duration_s_count"])
    grouped["peak_force_N_uncertainty"] = np.sqrt(u_random_f**2 + force_instrument_unc**2)

    return grouped
