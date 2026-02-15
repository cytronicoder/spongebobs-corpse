from __future__ import annotations

import pandas as pd

from .regression import adjacent_thickness_comparisons


def pairwise_adjacent_comparisons(summary_df: pd.DataFrame) -> pd.DataFrame:
    duration = adjacent_thickness_comparisons(
        summary_df,
        value_col="duration_s_mean",
        std_col="duration_s_std",
        count_col="duration_s_count",
        thickness_col="thickness_mm",
    )
    force = adjacent_thickness_comparisons(
        summary_df,
        value_col="peak_force_N_mean",
        std_col="peak_force_N_std",
        count_col="duration_s_count",
        thickness_col="thickness_mm",
    )

    rows: list[dict[str, object]] = []
    for variable, comps in (("contact_duration", duration), ("peak_force", force)):
        for comp in comps:
            rows.append(
                {
                    "variable": variable,
                    "group1": comp.group1,
                    "group2": comp.group2,
                    "mean_diff": comp.mean_diff,
                    "se_diff": comp.se_diff,
                    "t_statistic": comp.t_statistic,
                    "p_value": comp.p_value,
                    "significant": comp.significant,
                    "ci_lower": comp.ci_lower,
                    "ci_upper": comp.ci_upper,
                }
            )
    return pd.DataFrame(rows)
