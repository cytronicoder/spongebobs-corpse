from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.core import build_model_reports
from ..config.schema import AnalysisConfig
from ..io.load import load_batch_data
from ..io.paths import OutputManager
from ..io.save import save_figure, save_table, save_text
from ..processing import aggregate_by_thickness, validate_input_frame
from ..stats import fit_linear, fit_powerlaw, pairwise_adjacent_comparisons
from ..viz import axis_label, draw_cv_plot, draw_full_model_figure, draw_residual_plots


def _fit_models(summary_df: pd.DataFrame, method: str) -> list[dict]:
    x = summary_df["thickness_mm"].to_numpy(dtype=float)
    force_y = summary_df["peak_force_N_mean"].to_numpy(dtype=float)
    force_err = summary_df["peak_force_N_std"].to_numpy(dtype=float)
    tau_y = summary_df["duration_s_mean"].to_numpy(dtype=float)
    tau_err = summary_df["duration_s_std"].to_numpy(dtype=float)

    f_lin = fit_linear(x, force_y, yerr=force_err, method=method)
    t_lin = fit_linear(x, tau_y, yerr=tau_err, method=method)
    f_pow = fit_powerlaw(x, force_y, yerr=force_err, method=method)
    t_pow = fit_powerlaw(x, tau_y, yerr=tau_err, method=method)

    return [
        {
            "parameter": "peak_force",
            "model": "linear",
            **{k: v for k, v in f_lin.items() if k not in {"x", "y", "yerr", "residuals", "fitted_values", "x_original"}},
            "fitted_values": f_lin["fitted_values"],
            "residuals": f_lin["residuals"],
            "x": f_lin["x"],
            "y": f_lin["y"],
            "yerr": f_lin["yerr"],
        },
        {
            "parameter": "contact_duration",
            "model": "linear",
            **{k: v for k, v in t_lin.items() if k not in {"x", "y", "yerr", "residuals", "fitted_values", "x_original"}},
            "fitted_values": t_lin["fitted_values"],
            "residuals": t_lin["residuals"],
            "x": t_lin["x"],
            "y": t_lin["y"],
            "yerr": t_lin["yerr"],
        },
        {
            "parameter": "peak_force",
            "model": "powerlaw",
            **{k: v for k, v in f_pow.items() if k not in {"x", "y", "yerr", "residuals", "fitted_values", "x_original"}},
            "fitted_values": f_pow["fitted_values"],
            "residuals": f_pow["residuals"],
            "x": np.sqrt(f_pow["x_original"]),
            "y": f_pow["y"],
            "yerr": f_pow["yerr"],
        },
        {
            "parameter": "contact_duration",
            "model": "powerlaw",
            **{k: v for k, v in t_pow.items() if k not in {"x", "y", "yerr", "residuals", "fitted_values", "x_original"}},
            "fitted_values": t_pow["fitted_values"],
            "residuals": t_pow["residuals"],
            "x": np.sqrt(t_pow["x_original"]),
            "y": t_pow["y"],
            "yerr": t_pow["yerr"],
        },
    ]


def run_steps(config: AnalysisConfig, input_csv: Path, manager: OutputManager) -> list[str]:
    np.random.seed(config.analysis.seed)
    detailed = validate_input_frame(load_batch_data(input_csv))
    summary = aggregate_by_thickness(
        detailed,
        time_instrument_unc=config.analysis.random_time_uncertainty_s,
        force_instrument_unc=config.analysis.random_force_uncertainty_n,
    )

    detailed_artifact = save_table(detailed, manager.output_file("batch_detailed.csv"))
    summary_artifact = save_table(summary, manager.output_file("batch_summary.csv"))

    models = _fit_models(summary, config.analysis.regression_method)
    model_params = pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"x", "y", "yerr", "residuals", "fitted_values"}
            }
            for row in models
        ]
    )
    params_artifact = save_table(model_params, manager.output_file("model_parameters.csv"))

    residual_rows: list[dict[str, float | str]] = []
    for row in models:
        for xi, yi, fi, ri in zip(row["x"], row["y"], row["fitted_values"], row["residuals"]):
            residual_rows.append(
                {
                    "parameter": row["parameter"],
                    "model": row["model"],
                    "x": float(xi),
                    "observed": float(yi),
                    "fitted": float(fi),
                    "residual": float(ri),
                }
            )
    residual_artifact = save_table(pd.DataFrame(residual_rows), manager.output_file("model_residuals.csv"))

    comparisons = pairwise_adjacent_comparisons(summary)
    comparisons_artifact = save_table(comparisons, manager.output_file("pairwise_comparisons.csv"))

    summary_formatted = summary.copy()
    summary_formatted["duration_cv"] = summary_formatted["duration_cv"] * 100
    table = summary_formatted[
        [
            "thickness_mm",
            "duration_s_count",
            "duration_s_mean",
            "duration_s_std",
            "duration_s_uncertainty",
            "duration_cv",
            "peak_force_N_mean",
            "peak_force_N_std",
            "peak_force_N_uncertainty",
        ]
    ].rename(
        columns={
            "thickness_mm": "h (mm)",
            "duration_s_count": "n_trials",
            "duration_s_mean": "mean_tau (s)",
            "duration_s_std": "std_tau (s)",
            "duration_s_uncertainty": "unc_tau (s)",
            "duration_cv": "CV_tau (%)",
            "peak_force_N_mean": "mean_F_peak (N)",
            "peak_force_N_std": "std_F_peak (N)",
            "peak_force_N_uncertainty": "unc_F_peak (N)",
        }
    )
    summary_table_artifact = save_table(table, manager.output_file("summary_table_formatted.csv"))

    fit_summary_lines = [
        f"peak_force linear: s={models[0]['slope']:.4g} +/- {models[0]['slope_se']:.2g}, R^2={models[0]['r2']:.4f}",
        f"contact_duration linear: s={models[1]['slope']:.4g} +/- {models[1]['slope_se']:.2g}, R^2={models[1]['r2']:.4f}",
        f"peak_force powerlaw: b={models[2]['slope']:.4g} +/- {models[2]['slope_se']:.2g}, R^2={models[2]['r2']:.4f}",
        f"contact_duration powerlaw: b={models[3]['slope']:.4g} +/- {models[3]['slope_se']:.2g}, R^2={models[3]['r2']:.4f}",
        "",
        "Error bars: combined uncertainty",
        "u = sqrt(u_random^2 + u_instrument^2)",
    ]

    specs = [
        (models[0], axis_label("Thickness h", "mm"), axis_label("Peak force F_peak", "N"), "Linear Model", "blue"),
        (models[1], axis_label("Thickness h", "mm"), axis_label("Contact duration tau", "s"), "Linear Model", "orange"),
        (models[2], axis_label("sqrt(h)", "mm^0.5"), axis_label("Peak force F_peak", "N"), "Power-law Model", "blue"),
        (models[3], axis_label("sqrt(h)", "mm^0.5"), axis_label("Contact duration tau", "s"), "Power-law Model", "orange"),
    ]

    fig_main = draw_full_model_figure(specs, fit_summary_lines)
    save_figure(
        fig_main,
        stem="batch_analysis_plot",
        out_dir=manager.out_dir,
        dpi=config.io.figure_dpi,
        final_dir=manager.final_dir,
        caption_metadata={
            "data": "Mean peak force and mean contact duration versus pad thickness.",
            "uncertainty": "Error bars are combined uncertainty; shaded regions are 95% CI.",
            "method": config.analysis.regression_method,
            "n": int(summary["duration_s_count"].sum()),
            "note": "Legend and fit summaries are in the right-side gutter for print readability.",
        },
    )

    fig_cv = draw_cv_plot(summary)
    save_figure(
        fig_cv,
        stem="batch_cv_plot",
        out_dir=manager.out_dir,
        dpi=config.io.figure_dpi,
        final_dir=manager.final_dir,
        caption_metadata={
            "data": "Coefficient of variation of contact duration by thickness.",
            "uncertainty": "CV = SD/mean.",
            "method": "Descriptive repeatability analysis",
            "n": int(summary["duration_s_count"].sum()),
        },
    )

    fig_res = draw_residual_plots(models)
    save_figure(
        fig_res,
        stem="residual_plots",
        out_dir=manager.out_dir,
        dpi=config.io.figure_dpi,
        final_dir=manager.final_dir,
        caption_metadata={
            "data": "Residuals versus fitted values for each regression model.",
            "uncertainty": "Residuals from fitted models.",
            "method": config.analysis.regression_method,
            "n": len(models),
        },
    )

    model_report, analysis_summary = build_model_reports(model_params)
    report_artifact = save_text(model_report, manager.output_file("model_comparison.txt"))
    stat_report_artifact = save_text(
        "ADVANCED STATISTICAL ANALYSIS REPORT\n" + "=" * 80 + "\n\n" + model_report,
        manager.output_file("statistical_report.txt"),
    )
    additional_artifact = save_text(
        "ADDITIONAL STATISTICS\n" + "=" * 80 + "\n" + comparisons.to_string(index=False),
        manager.output_file("additional_statistics.txt"),
    )
    summary_artifact_txt = save_text(analysis_summary, manager.output_file("ANALYSIS_SUMMARY.txt"))

    # (Legacy top-level deliverables removed.)

    artifacts = [
        str(detailed_artifact),
        str(summary_artifact),
        str(params_artifact),
        str(residual_artifact),
        str(comparisons_artifact),
        str(summary_table_artifact),
        str(report_artifact),
        str(stat_report_artifact),
        str(additional_artifact),
        str(summary_artifact_txt),
    ]
    return artifacts
