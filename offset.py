"""
Aligns multiple experimental runs recorded in a single CSV by shifting
each run's time axis so that the absolute force peak coincides with the
reference run's peak.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS = [
    "Time (s)",
    "Force (N)",
    "Position (m)",
    "Velocity (m/s)",
    "Acceleration (m/s²)",
]


def detect_runs(columns: List[str]) -> List[str]:
    """
    Infer run label prefixes from column headers and return them in a sensible order.
    Numbered runs appear first in numeric order, followed by "Latest", then any others.
    """
    prefixes = []
    for col in columns:
        if ":" in col:
            prefixes.append(col.split(":")[0].strip())
    uniq = sorted(set(prefixes))

    def sort_key(p: str) -> Tuple[int, int, str]:
        if p.startswith("Run "):
            m = re.match(r"Run\s+(\d+)", p)
            if m:
                return (0, int(m.group(1)), "")
            return (0, 10_000, p)
        if p == "Latest":
            return (1, 0, "")
        return (2, 0, p)

    return sorted(uniq, key=sort_key)


def extract_run_series(df: pd.DataFrame, run_label: str) -> Dict[str, pd.Series]:
    """
    Extract metric series for a given run label.
    Missing columns are returned as NaN Series.
    """
    out = {}
    for metric in METRICS:
        col = f"{run_label}: {metric}"
        if col in df.columns:
            out[metric] = df[col]
        else:
            out[metric] = pd.Series([np.nan] * len(df))
    return out


def absolute_force_peak(force_s: pd.Series, time_s: pd.Series) -> Tuple[int, float, float]:
    """
    Return (row_index, t_peak, f_peak) for the absolute maximum of |force|.
    If no valid data exist, returns (-1, nan, nan).
    """
    mask = force_s.notna() & time_s.notna()
    if not mask.any():
        return (-1, np.nan, np.nan)
    idx = (force_s[mask].abs()).idxmax()
    return int(idx), float(time_s.loc[idx]), float(force_s.loc[idx])


def first_valid_value(s: pd.Series) -> float:
    """
    Return the first non-NaN value from a Series or NaN if none exist.
    """
    fv = s[s.notna()]
    return float(fv.iloc[0]) if len(fv) else np.nan


def align_runs_by_abs_force_peak(
    csv_path: Path,
    reference_run: str = "Run 1",
    save_aligned_csv: bool = True,
    output_dir: Path = None,
    encoding: str = "utf-8-sig"
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Align runs in a CSV by absolute Force peak time relative to a reference run.
    """
    df = pd.read_csv(csv_path, encoding=encoding)
    runs = detect_runs(list(df.columns))
    if reference_run not in runs:
        raise ValueError(
            f"Reference run '{reference_run}' not found. Runs detected: {runs}")

    run_data = {run: extract_run_series(df, run) for run in runs}
    _, t_ref, f_ref = absolute_force_peak(run_data[reference_run]["Force (N)"],
                                          run_data[reference_run]["Time (s)"])
    if not np.isfinite(t_ref):
        raise ValueError(
            f"Could not determine peak time for reference run '{reference_run}'.")

    offsets = []
    for run in runs:
        _, t_peak, f_peak = absolute_force_peak(
            run_data[run]["Force (N)"], run_data[run]["Time (s)"])
        dt = (t_peak - t_ref) if np.isfinite(t_peak) else np.nan
        offsets.append({
            "Run": run,
            "Peak |Force| (N)": abs(f_peak) if np.isfinite(f_peak) else np.nan,
            "Time at Peak (s)": t_peak,
            "Offset dt (s) vs reference": dt,
            "Applied Shift (s)": -dt if np.isfinite(dt) else np.nan
        })
    offsets_df = pd.DataFrame(offsets)

    aligned_blocks = {}
    start_times_after_shift = []
    for run in runs:
        dt = offsets_df.loc[offsets_df["Run"] == run,
                            "Offset dt (s) vs reference"].values[0]
        if np.isfinite(dt):
            t_aligned = run_data[run]["Time (s)"] - dt
        else:
            t_aligned = run_data[run]["Time (s)"]
        start_times_after_shift.append(first_valid_value(t_aligned))
        aligned_blocks[(run, "Time (s) [aligned]")] = t_aligned
        aligned_blocks[(run, "Force (N)")] = run_data[run]["Force (N)"]
        aligned_blocks[(run, "Position (m)")] = run_data[run]["Position (m)"]
        aligned_blocks[(run, "Velocity (m/s)")
                       ] = run_data[run]["Velocity (m/s)"]
        aligned_blocks[(run, "Acceleration (m/s²)")
                       ] = run_data[run]["Acceleration (m/s²)"]

    left_cut = float(np.nanmax(start_times_after_shift))

    masks = {}
    for run in runs:
        masks[run] = aligned_blocks[(run, "Time (s) [aligned]")] >= left_cut

    for run in runs:
        mask = masks[run]
        for metric in ["Time (s) [aligned]", "Force (N)", "Position (m)", "Velocity (m/s)", "Acceleration (m/s²)"]:
            series = aligned_blocks[(run, metric)]
            aligned_blocks[(run, metric)] = series[mask].reset_index(drop=True)

    aligned_df = pd.concat(aligned_blocks, axis=1)

    if output_dir is None:
        output_dir = csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_aligned_csv:
        aligned_name = csv_path.with_suffix("").name + "_aligned.csv"
        aligned_path = output_dir / aligned_name
        aligned_df.to_csv(aligned_path, index=False)

    offsets_name = csv_path.with_suffix("").name + "_offsets.csv"
    offsets_path = output_dir / offsets_name
    offsets_df.to_csv(offsets_path, index=False)

    return aligned_df, offsets_df, left_cut


def plot_metric_overlay(aligned_df: pd.DataFrame, metric: str, out_dir: Path, title_suffix: str = "") -> Path:
    """
    Create an overlay plot for a given metric versus aligned time across all runs.
    Returns the saved PNG path.
    """
    runs = sorted(set([c[0] for c in aligned_df.columns]))
    fig, ax = plt.subplots()
    for run in runs:
        t = aligned_df[(run, "Time (s) [aligned]")].values
        y = aligned_df[(run, metric)].values
        mask = np.isfinite(t) & np.isfinite(y)
        if mask.any():
            ax.plot(t[mask], y[mask], label=run)
    ax.set_xlabel("Time (s) [aligned]")
    ax.set_ylabel(metric)
    ttl = f"{metric} vs Time [Aligned]"
    if title_suffix:
        ttl = f"{ttl} — {title_suffix}"
    ax.set_title(ttl)
    ax.legend()
    ax.grid(True)
    out_path = out_dir / \
        f"overlay_{metric.replace('/', '_').replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_metric_per_run(aligned_df: pd.DataFrame, metric: str, out_dir: Path, title_suffix: str = "") -> List[Path]:
    """
    Create one figure per run for a given metric versus aligned time.
    Returns a list of saved PNG paths.
    """
    saved = []
    runs = sorted(set([c[0] for c in aligned_df.columns]))
    for run in runs:
        t = aligned_df[(run, "Time (s) [aligned]")].values
        y = aligned_df[(run, metric)].values
        mask = np.isfinite(t) & np.isfinite(y)
        if not mask.any():
            continue
        fig, ax = plt.subplots()
        ax.plot(t[mask], y[mask], label=f"{run} — {metric}")
        ax.set_xlabel("Time (s) [aligned]")
        ax.set_ylabel(metric)
        ttl = f"{metric} vs Time [Aligned] — {run}"
        if title_suffix:
            ttl = f"{ttl} — {title_suffix}"
        ax.set_title(ttl)
        ax.legend()
        ax.grid(True)
        out_path = out_dir / \
            f"{run.replace(' ', '')}_{metric.replace('/', '_').replace(' ', '_')}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
    return saved


def process_csv(
    csv_path: Path,
    reference_run: str,
    outputs_root: Path,
    save_aligned_csv: bool,
    show_plots: bool
) -> None:
    """
    Process a single CSV: align runs, save aligned artifacts, and generate all time-based plots.
    """
    file_stem = csv_path.with_suffix("").name
    file_out_dir = outputs_root / file_stem
    file_out_dir.mkdir(parents=True, exist_ok=True)

    aligned_df, offsets_df, left_cut = align_runs_by_abs_force_peak(
        csv_path=csv_path,
        reference_run=reference_run,
        save_aligned_csv=save_aligned_csv,
        output_dir=file_out_dir
    )

    print(f"\n=== File: {csv_path.name} ===")
    print(offsets_df.to_string(index=False))
    print(f"Left cut applied at t = {left_cut:.6f} s")

    overlay_dir = file_out_dir / "overlays"
    perrun_dir = file_out_dir / "per_run"
    overlay_dir.mkdir(exist_ok=True)
    perrun_dir.mkdir(exist_ok=True)

    for metric in ["Force (N)", "Position (m)", "Velocity (m/s)", "Acceleration (m/s²)"]:
        plot_metric_overlay(aligned_df, metric, overlay_dir,
                            title_suffix=file_stem)
        plot_metric_per_run(aligned_df, metric, perrun_dir,
                            title_suffix=file_stem)

    if show_plots:
        for metric in ["Force (N)", "Position (m)", "Velocity (m/s)", "Acceleration (m/s²)"]:
            runs = sorted(set([c[0] for c in aligned_df.columns]))
            fig, ax = plt.subplots()
            for run in runs:
                t = aligned_df[(run, "Time (s) [aligned]")].values
                y = aligned_df[(run, metric)].values
                mask = np.isfinite(t) & np.isfinite(y)
                if mask.any():
                    ax.plot(t[mask], y[mask], label=run)
            ax.set_xlabel("Time (s) [aligned]")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} vs Time [Aligned] — {file_stem}")
            ax.legend()
            ax.grid(True)
            plt.show()


def main() -> None:
    """
    Batch process all CSVs in an input directory and write artifacts to an output directory.

    --input_dir : directory containing *.csv files to process (default: ./data)
    --output_dir: directory to store aligned CSVs and plots (default: ./outputs)
    --reference_run: run label used for alignment baseline (default: "Run 1")
    --no_save_aligned: disable writing aligned CSVs
    --show: display plots interactively in addition to saving PNGs
    """
    parser = argparse.ArgumentParser(
        description="Align runs by absolute Force peak and plot time-series.")
    parser.add_argument("--input_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--reference_run", type=str, default="Run 1")
    parser.add_argument("--no_save_aligned", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(list(input_dir.glob("*.csv")))
    if not csv_files:
        print(f"No CSV files found in {input_dir.resolve()}.")
        return

    for csv_path in csv_files:
        try:
            process_csv(
                csv_path=csv_path,
                reference_run=args.reference_run,
                outputs_root=output_dir,
                save_aligned_csv=not args.no_save_aligned,
                show_plots=args.show
            )
        except Exception as exc:
            print(f"Failed to process {csv_path.name}: {exc}")


if __name__ == "__main__":
    main()
