"""
Contact Duration Analysis for Impact Experiments

Analyzes contact duration from aligned force-time data to investigate
the relationship between pad thickness and contact duration during
controlled cart-into-pad impacts.

The script implements multiple contact detection methods:
1. Force threshold method (primary)
2. Velocity-based method
3. Energy-based method

Usage:
    python contact_duration.py --input_dir outputs --threshold 0.05
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
    
    Examples:
        "dr lee go brr - 4mm" -> 4.0
        "dr lee go brr - 28mm" -> 28.0
    """
    match = re.search(r'(\d+(?:\.\d+)?)\s*mm', filename, re.IGNORECASE)
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


def smooth_signal(data: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
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
    
    # Handle NaN values
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
    smooth: bool = True
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
    # Filter valid data
    mask = np.isfinite(time) & np.isfinite(force)
    if not mask.any():
        return {
            'duration': np.nan,
            'start_time': np.nan,
            'end_time': np.nan,
            'peak_force': np.nan,
            'threshold_force': np.nan,
            'method': 'threshold'
        }
    
    t_valid = time[mask]
    f_valid = force[mask]
    
    # Smooth force signal if requested
    if smooth:
        f_valid = smooth_signal(f_valid)
    
    # Calculate threshold based on peak absolute force
    abs_force = np.abs(f_valid)
    peak_force = np.max(abs_force)
    threshold = threshold_fraction * peak_force
    
    # Find peak index
    peak_idx = np.argmax(abs_force)
    
    # Find contact start: first point before peak exceeding threshold
    start_idx = 0
    for i in range(peak_idx, -1, -1):
        if abs_force[i] < threshold:
            start_idx = i + 1 if i + 1 < len(abs_force) else i
            break
    
    # Find contact end: first point after peak dropping below threshold
    end_idx = len(abs_force) - 1
    for i in range(peak_idx, len(abs_force)):
        if abs_force[i] < threshold:
            end_idx = i
            break
    
    # Calculate duration
    start_time = t_valid[start_idx]
    end_time = t_valid[end_idx]
    duration = end_time - start_time
    
    return {
        'duration': duration,
        'start_time': start_time,
        'end_time': end_time,
        'peak_force': peak_force,
        'threshold_force': threshold,
        'method': 'threshold'
    }


def calculate_contact_duration_velocity(
    time: np.ndarray,
    velocity: np.ndarray,
    smooth: bool = True
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
            'duration': np.nan,
            'start_time': np.nan,
            'end_time': np.nan,
            'min_velocity': np.nan,
            'method': 'velocity'
        }
    
    t_valid = time[mask]
    v_valid = velocity[mask]
    
    if smooth:
        v_valid = smooth_signal(v_valid)
    
    # Find minimum velocity (maximum deceleration)
    min_vel_idx = np.argmin(v_valid)
    min_vel = v_valid[min_vel_idx]
    
    # Find where velocity goes negative before minimum
    start_idx = 0
    for i in range(min_vel_idx, -1, -1):
        if v_valid[i] >= 0:
            start_idx = i + 1 if i + 1 < len(v_valid) else i
            break
    
    # Find where velocity returns to zero/positive after minimum
    end_idx = len(v_valid) - 1
    for i in range(min_vel_idx, len(v_valid)):
        if v_valid[i] >= 0:
            end_idx = i
            break
    
    start_time = t_valid[start_idx]
    end_time = t_valid[end_idx]
    duration = end_time - start_time
    
    return {
        'duration': duration,
        'start_time': start_time,
        'end_time': end_time,
        'min_velocity': min_vel,
        'method': 'velocity'
    }


def analyze_contact_duration_for_file(
    aligned_csv_path: Path,
    threshold_fraction: float = 0.05,
    smooth: bool = True,
    method: str = 'threshold'
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
    # Read aligned data with multi-index columns
    df = pd.read_csv(aligned_csv_path, header=[0, 1])
    runs = detect_runs(df.columns)
    
    # Extract pad thickness from filename
    thickness = extract_thickness_from_filename(aligned_csv_path.stem)
    
    results = []
    
    for run in runs:
        try:
            time = df[(run, 'Time (s) [aligned]')].values
            force = df[(run, 'Force (N)')].values
            
            # Threshold method
            if method in ['threshold', 'both']:
                metrics_threshold = calculate_contact_duration_threshold(
                    time, force, threshold_fraction, smooth
                )
                
                result = {
                    'file': aligned_csv_path.stem,
                    'thickness_mm': thickness,
                    'run': run,
                    'duration_s': metrics_threshold['duration'],
                    'start_time_s': metrics_threshold['start_time'],
                    'end_time_s': metrics_threshold['end_time'],
                    'peak_force_N': metrics_threshold['peak_force'],
                    'threshold_force_N': metrics_threshold['threshold_force'],
                    'method': 'threshold'
                }
                results.append(result)
            
            # Velocity method
            if method in ['velocity', 'both']:
                if (run, 'Velocity (m/s)') in df.columns:
                    velocity = df[(run, 'Velocity (m/s)')].values
                    metrics_velocity = calculate_contact_duration_velocity(
                        time, velocity, smooth
                    )
                    
                    result = {
                        'file': aligned_csv_path.stem,
                        'thickness_mm': thickness,
                        'run': run,
                        'duration_s': metrics_velocity['duration'],
                        'start_time_s': metrics_velocity['start_time'],
                        'end_time_s': metrics_velocity['end_time'],
                        'min_velocity_m_s': metrics_velocity['min_velocity'],
                        'method': 'velocity'
                    }
                    results.append(result)
        
        except Exception as e:
            print(f"Warning: Failed to process {run} in {aligned_csv_path.name}: {e}")
            continue
    
    return pd.DataFrame(results)


def aggregate_by_thickness(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate contact duration results by pad thickness.
    
    Calculates mean, std, min, max, and count for each thickness.
    """
    # Filter out NaN durations
    valid_df = results_df[results_df['duration_s'].notna()].copy()
    
    if len(valid_df) == 0:
        return pd.DataFrame()
    
    # Group by thickness and method
    agg_funcs = {
        'duration_s': ['mean', 'std', 'min', 'max', 'count'],
        'peak_force_N': ['mean', 'std'] if 'peak_force_N' in valid_df.columns else []
    }
    
    # Remove empty aggregations
    agg_funcs = {k: v for k, v in agg_funcs.items() if v}
    
    grouped = valid_df.groupby(['thickness_mm', 'method']).agg(agg_funcs)
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    # Calculate coefficient of variation
    if 'duration_s_mean' in grouped.columns and 'duration_s_std' in grouped.columns:
        grouped['duration_cv'] = grouped['duration_s_std'] / grouped['duration_s_mean']
    
    return grouped.sort_values('thickness_mm')


def perform_regression_analysis(aggregated_df: pd.DataFrame, method: str = 'threshold') -> Dict:
    """
    Perform linear regression: contact duration vs. pad thickness.
    
    Returns regression statistics and model parameters.
    """
    # Filter by method
    method_df = aggregated_df[aggregated_df['method'] == method].copy()
    
    if len(method_df) < 2:
        return {
            'slope': np.nan,
            'intercept': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'std_err': np.nan,
            'n_points': len(method_df)
        }
    
    x = method_df['thickness_mm'].values
    y = method_df['duration_s_mean'].values
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err,
        'n_points': len(method_df),
        'equation': f"Duration = {slope:.6f} * Thickness + {intercept:.6f}"
    }


def plot_duration_vs_thickness(
    results_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    output_path: Path,
    method: str = 'threshold'
):
    """
    Create publication-quality plot of contact duration vs. pad thickness.
    """
    # Filter by method
    method_results = results_df[results_df['method'] == method].copy()
    method_agg = aggregated_df[aggregated_df['method'] == method].copy()
    
    if len(method_agg) == 0:
        print(f"No data available for {method} method")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual runs as scatter points
    for thickness in method_results['thickness_mm'].unique():
        if pd.isna(thickness):
            continue
        thickness_data = method_results[method_results['thickness_mm'] == thickness]
        ax.scatter(
            [thickness] * len(thickness_data),
            thickness_data['duration_s'].values,
            alpha=0.3,
            s=50,
            color='gray',
            label='Individual runs' if thickness == method_results['thickness_mm'].unique()[0] else ''
        )
    
    # Plot means with error bars
    x = method_agg['thickness_mm'].values
    y = method_agg['duration_s_mean'].values
    yerr = method_agg['duration_s_std'].values if 'duration_s_std' in method_agg.columns else None
    
    ax.errorbar(
        x, y, yerr=yerr,
        fmt='o-',
        markersize=10,
        linewidth=2,
        capsize=5,
        capthick=2,
        label='Mean ± SD',
        color='#2E86AB',
        ecolor='#2E86AB'
    )
    
    # Perform and plot regression
    reg_stats = perform_regression_analysis(aggregated_df, method)
    
    if not np.isnan(reg_stats['slope']):
        x_fit = np.linspace(x.min() * 0.9, x.max() * 1.1, 100)
        y_fit = reg_stats['slope'] * x_fit + reg_stats['intercept']
        ax.plot(x_fit, y_fit, '--', color='#A23B72', linewidth=2, label='Linear fit')
        
        # Add regression equation and R² to plot
        textstr = f"$y = {reg_stats['slope']:.6f}x + {reg_stats['intercept']:.6f}$\n"
        textstr += f"$R^2 = {reg_stats['r_squared']:.4f}$\n"
        textstr += f"$p = {reg_stats['p_value']:.4f}$"
        
        ax.text(
            0.05, 0.95,
            textstr,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11
        )
    
    ax.set_xlabel('Pad Thickness (mm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Contact Duration (s)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Contact Duration vs. Pad Thickness\n({method.capitalize()} Method)',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_path}")


def plot_annotated_force_profiles(
    aligned_csv_path: Path,
    contact_results: pd.DataFrame,
    output_dir: Path,
    method: str = 'threshold'
):
    """
    Plot force-time profiles with contact duration windows annotated.
    """
    df = pd.read_csv(aligned_csv_path, header=[0, 1])
    runs = detect_runs(df.columns)
    thickness = extract_thickness_from_filename(aligned_csv_path.stem)
    
    # Filter contact results for this file and method
    file_results = contact_results[
        (contact_results['file'] == aligned_csv_path.stem) &
        (contact_results['method'] == method)
    ]
    
    # Create overlay plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for run in runs:
        try:
            time = df[(run, 'Time (s) [aligned]')].values
            force = df[(run, 'Force (N)')].values
            
            mask = np.isfinite(time) & np.isfinite(force)
            if not mask.any():
                continue
            
            # Plot force profile
            ax.plot(time[mask], force[mask], label=run, linewidth=1.5, alpha=0.7)
            
            # Get contact window for this run
            run_contact = file_results[file_results['run'] == run]
            if len(run_contact) > 0:
                start = run_contact.iloc[0]['start_time_s']
                end = run_contact.iloc[0]['end_time_s']
                
                if np.isfinite(start) and np.isfinite(end):
                    # Shade contact duration window
                    ax.axvspan(start, end, alpha=0.1, color='red')
                    
                    # Add vertical lines at start/end
                    ax.axvline(start, color='green', linestyle='--', alpha=0.5, linewidth=1)
                    ax.axvline(end, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        except Exception as e:
            print(f"Warning: Could not plot {run}: {e}")
            continue
    
    ax.set_xlabel('Time (s) [aligned]', fontsize=12)
    ax.set_ylabel('Force (N)', fontsize=12)
    ax.set_title(
        f'Force Profiles with Contact Duration Windows\n{aligned_csv_path.stem} (Thickness: {thickness} mm)',
        fontsize=13,
        fontweight='bold'
    )
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    output_path = output_dir / f"{aligned_csv_path.stem}_contact_annotated.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved annotated plot: {output_path}")


def main():
    """
    Main analysis pipeline for contact duration analysis.
    """
    parser = argparse.ArgumentParser(
        description="Analyze contact duration from aligned impact data."
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='outputs',
        help='Directory containing aligned CSV files (default: outputs)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='contact_analysis',
        help='Directory to save analysis results (default: contact_analysis)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.05,
        help='Force threshold fraction for contact detection (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='threshold',
        choices=['threshold', 'velocity', 'both'],
        help='Contact detection method (default: threshold)'
    )
    parser.add_argument(
        '--no_smooth',
        action='store_true',
        help='Disable signal smoothing'
    )
    parser.add_argument(
        '--plots',
        action='store_true',
        help='Generate annotated force profile plots'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("CONTACT DURATION ANALYSIS")
    print("="*70)
    print(f"Input directory: {input_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Detection method: {args.method}")
    print(f"Force threshold: {args.threshold * 100:.1f}% of peak")
    print(f"Signal smoothing: {'Disabled' if args.no_smooth else 'Enabled'}")
    print("="*70)
    
    # Find all aligned CSV files
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
    
    print(f"\nFound {len(aligned_files)} aligned CSV files:")
    for f in aligned_files:
        thickness = extract_thickness_from_filename(f.stem)
        print(f"  - {f.name} (thickness: {thickness} mm)")
    
    # Analyze each file
    all_results = []
    
    for aligned_csv in aligned_files:
        print(f"\nProcessing: {aligned_csv.name}")
        try:
            results = analyze_contact_duration_for_file(
                aligned_csv,
                threshold_fraction=args.threshold,
                smooth=not args.no_smooth,
                method=args.method
            )
            all_results.append(results)
            print(f"  Analyzed {len(results)} runs")
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if not all_results:
        print("\nNo results generated. Exiting.")
        return
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save detailed results
    detailed_path = output_dir / "contact_duration_detailed.csv"
    combined_results.to_csv(detailed_path, index=False)
    print(f"\nSaved detailed results: {detailed_path}")
    
    # Aggregate by thickness
    aggregated = aggregate_by_thickness(combined_results)
    
    if len(aggregated) == 0:
        print("\nNo valid data for aggregation.")
        return
    
    # Save aggregated results
    aggregated_path = output_dir / "contact_duration_summary.csv"
    aggregated.to_csv(aggregated_path, index=False)
    print(f"Saved summary statistics: {aggregated_path}")
    
    # Display summary
    print("\n" + "="*70)
    print("SUMMARY: Contact Duration by Pad Thickness")
    print("="*70)
    print(aggregated.to_string(index=False))
    
    # Perform regression analysis
    print("\n" + "="*70)
    print("REGRESSION ANALYSIS")
    print("="*70)
    
    methods_to_analyze = ['threshold'] if args.method == 'threshold' else ['threshold', 'velocity'] if args.method == 'both' else ['velocity']
    
    for method in methods_to_analyze:
        reg_stats = perform_regression_analysis(aggregated, method)
        print(f"\n{method.capitalize()} Method:")
        print(f"  Equation: {reg_stats.get('equation', 'N/A')}")
        print(f"  R² = {reg_stats['r_squared']:.4f}")
        print(f"  p-value = {reg_stats['p_value']:.4f}")
        print(f"  Standard error = {reg_stats['std_err']:.6f}")
        print(f"  Data points = {reg_stats['n_points']}")
        
        # Interpret results
        if reg_stats['p_value'] < 0.05:
            print(f"  → Statistically significant relationship (p < 0.05)")
        else:
            print(f"  → No significant relationship (p >= 0.05)")
        
        # Generate plot
        plot_path = output_dir / f"contact_duration_vs_thickness_{method}.png"
        plot_duration_vs_thickness(
            combined_results,
            aggregated,
            plot_path,
            method=method
        )
    
    # Generate annotated force profiles if requested
    if args.plots:
        print("\n" + "="*70)
        print("GENERATING ANNOTATED FORCE PROFILES")
        print("="*70)
        
        plots_dir = output_dir / "annotated_profiles"
        plots_dir.mkdir(exist_ok=True)
        
        for aligned_csv in aligned_files:
            try:
                plot_annotated_force_profiles(
                    aligned_csv,
                    combined_results,
                    plots_dir,
                    method='threshold'  # Use primary method
                )
            except Exception as e:
                print(f"Warning: Could not create plot for {aligned_csv.name}: {e}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
