from __future__ import annotations

import pandas as pd


def build_model_reports(model_params_df: pd.DataFrame) -> tuple[str, str]:
    lines = [
        "REGRESSION ANALYSIS WITH PARAMETER UNCERTAINTIES",
        "=" * 80,
        "",
    ]

    for _, row in model_params_df.iterrows():
        lines.append(f"--- {row['parameter']} ({row['model']}) ---")
        lines.append(f"  slope: {row['slope']:.6f} +/- {row['slope_se']:.6f}")
        lines.append(f"  slope 95% CI: [{row['slope_ci_lower']:.6f}, {row['slope_ci_upper']:.6f}]")
        lines.append(f"  slope p-value: {row['p_value_slope']:.6g}")
        lines.append(f"  intercept: {row['intercept']:.6f} +/- {row['intercept_se']:.6f}")
        ic_low, ic_high = row["intercept_ci_lower"], row["intercept_ci_upper"]
        lines.append(f"  intercept 95% CI: [{ic_low:.6f}, {ic_high:.6f}]")
        lines.append(f"  R^2: {row['r2']:.4f}")
        lines.append(f"  Adj. R^2: {row['adj_r2']:.4f}")
        lines.append("")

    model_report = "\n".join(lines).strip() + "\n"

    summary_lines = [
        "BATCH ANALYSIS SUMMARY",
        "=" * 80,
        f"model rows: {len(model_params_df)}",
        "",
        "Best-fit model R^2 values:",
    ]
    for _, row in model_params_df.iterrows():
        p, m, r2, ar2 = row["parameter"], row["model"], row["r2"], row["adj_r2"]
        summary_lines.append(f"  {p} [{m}]: R^2={r2:.4f}, adj R^2={ar2:.4f}")
    return model_report, "\n".join(summary_lines).strip() + "\n"
