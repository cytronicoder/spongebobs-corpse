# Scripts Documentation

## Overview

This project includes three main Python scripts:

1. **analysis.py** - Statistical analysis and visualization
2. **contact_duration.py** - Contact duration calculation from force-time data
3. **utils.py** - Shared utility functions

## analysis.py

### Purpose
Performs comprehensive statistical analysis on contact duration and peak force data, including linear regression, uncertainty analysis, and visualization.

### Usage

```bash
python analysis.py
```

**Note**: Script automatically searches for aligned data files in the `outputs/` directory.

### Input
- Aligned CSV files located in `outputs/dr lee go brr - <thickness>mm/` directories
- Contact duration summary files from prior analysis

### Output
Generated in `outputs/contact_analysis_<threshold>/`:
- `contact_duration_summary.csv` - Summary statistics
- `summary_table_formatted.csv` - LaTeX-formatted table
- `statistical_report.txt` - Detailed regression analysis
- `annotated_profiles/` - Force-time plots with contact regions highlighted

### Key Functions

#### `plot_duration_subplot(ax, detailed_df, summary_df, colors)`
Creates scatter plot of contact duration vs. thickness with regression analysis.

**Parameters**:
- `ax`: Matplotlib axis object
- `detailed_df`: DataFrame with individual run measurements
- `summary_df`: DataFrame with summary statistics
- `colors`: Dictionary of color codes for plot elements

**Visualization includes**:
- Individual data points with jitter for visibility
- Mean values with error bars (±1 standard deviation)
- Best-fit linear regression line
- 95% confidence interval band
- Min/max slope uncertainty bounds
- Regression statistics annotation

#### `plot_peak_force_subplot(ax, detailed_df, summary_df, colors)`
Creates scatter plot of peak force vs. thickness with regression analysis.

**Parameters**: Same as `plot_duration_subplot`

**Visualization**: Similar to duration plot but for peak force data

#### `plot_cv_subplot(ax, summary_df, colors, y_variable)`
Plots coefficient of variation (CV) vs. thickness.

**Parameters**:
- `ax`: Matplotlib axis object
- `summary_df`: DataFrame with summary statistics
- `colors`: Dictionary of color codes
- `y_variable`: Column name for CV data (`duration_s_cv` or `peak_force_N_cv`)

**Features**:
- Bar plot or scatter plot visualization
- Horizontal reference line at typical precision threshold
- Automatic y-axis scaling

#### `create_regression_text(slope, intercept, r_value, p_value, slope_unc, intercept_unc, var_name)`
Generates formatted text box with regression statistics.

**Parameters**:
- `slope`: Regression slope coefficient
- `intercept`: Y-intercept
- `r_value`: Correlation coefficient
- `p_value`: Statistical significance
- `slope_unc`: Uncertainty in slope (from bootstrap)
- `intercept_unc`: Uncertainty in intercept (from bootstrap)
- `var_name`: Variable name for labeling (e.g., "duration")

**Returns**: Formatted string with LaTeX-style equations and statistical measures

#### `generate_statistical_report(detailed_df, summary_df, output_dir)`
Creates comprehensive text report of statistical analysis.

**Parameters**:
- `detailed_df`: Individual measurements
- `summary_df`: Summary statistics
- `output_dir`: Directory path for output file

**Output**: `statistical_report.txt` containing:
- Linear regression results for duration vs. thickness
- Linear regression results for peak force vs. thickness
- Coefficient of variation analysis
- Statistical significance assessments

### Configuration

Modify these variables in the script to customize analysis:

```python
# Bootstrap parameters
n_boot = 1000  # Number of bootstrap iterations

# Plot aesthetics
colors = {
    "gray": "#7f8c8d",
    "blue": "#3498db",
    "orange": "#e67e22",
    "pink": "#e74c3c"
}

# Figure size
fig_width = 18  # inches
fig_height = 6  # inches

# Font sizes
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
```

## contact_duration.py

### Purpose
Calculates contact duration from aligned force-time data using multiple detection methods.

### Usage

```bash
python contact_duration.py <aligned_csv_file> [options]
```

### Command-Line Arguments

- `aligned_csv_file` (required): Path to aligned CSV file
- `--threshold` (optional): Force threshold fraction (default: 0.05)
  - Example: `--threshold 0.03` for 3% of peak force
- `--method` (optional): Detection method (default: threshold)
  - Options: `threshold`, `velocity`, `energy`
- `--smooth` (optional): Enable signal smoothing (default: True)
  - Use `--no-smooth` to disable

### Examples

```bash
# Basic usage with 5% threshold
python contact_duration.py outputs/dr\ lee\ go\ brr\ -\ 4mm/dr\ lee\ go\ brr\ -\ 4mm_aligned.csv

# Custom threshold
python contact_duration.py outputs/dr\ lee\ go\ brr\ -\ 4mm/dr\ lee\ go\ brr\ -\ 4mm_aligned.csv --threshold 0.03

# Different method
python contact_duration.py outputs/dr\ lee\ go\ brr\ -\ 4mm/dr\ lee\ go\ brr\ -\ 4mm_aligned.csv --method velocity

# Disable smoothing
python contact_duration.py outputs/dr\ lee\ go\ brr\ -\ 4mm/dr\ lee\ go\ brr\ -\ 4mm_aligned.csv --no-smooth
```

### Input
- Aligned CSV file with multi-level columns
- Each run must have `time_s` and `force_N` columns
- Optional: `velocity_m/s` for velocity method, `energy_J` for energy method

### Output
Generated in `outputs/contact_analysis_<threshold>pct/`:
- `contact_duration_detailed.csv` - Run-by-run measurements
- `contact_duration_summary.csv` - Summary statistics per thickness
- `summary_table_formatted.csv` - LaTeX-formatted results
- `annotated_profiles/<thickness>_<run>.png` - Annotated force-time plots

### Key Functions

#### `extract_thickness_from_filename(filename: str) -> Optional[float]`
Extracts pad thickness from filename using regex pattern.

**Parameters**:
- `filename`: String containing thickness information

**Returns**: Float thickness in mm, or None if not found

**Example**:
```python
thickness = extract_thickness_from_filename("experiment - 40mm.csv")
# Returns: 40.0
```

#### `detect_runs(columns: pd.MultiIndex) -> List[str]`
Identifies unique run labels from multi-level column headers.

**Parameters**:
- `columns`: Pandas MultiIndex from aligned DataFrame

**Returns**: Sorted list of run identifiers

**Example output**: `['Run 1', 'Run 2', 'Run 3', 'Latest']`

#### `smooth_signal(data: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray`
Applies Savitzky-Golay filter to reduce noise in force signals.

**Parameters**:
- `data`: Input signal array
- `window_length`: Filter window size (must be odd, default: 11)
- `polyorder`: Polynomial order for fitting (default: 3)

**Returns**: Smoothed signal array

**Notes**:
- Preserves peak characteristics while reducing high-frequency noise
- Handles NaN values by masking
- Returns original data if insufficient points for filtering

#### `calculate_contact_duration_threshold(time, force, threshold_fraction=0.05, smooth=True) -> Dict`
Primary method: Calculates contact duration using force threshold.

**Parameters**:
- `time`: Time array (seconds)
- `force`: Force array (Newtons)
- `threshold_fraction`: Fraction of peak force for threshold (default: 0.05)
- `smooth`: Whether to apply smoothing filter (default: True)

**Returns**: Dictionary with:
- `duration`: Contact duration (s)
- `start_time`: Contact start time (s)
- `end_time`: Contact end time (s)
- `peak_force`: Maximum absolute force (N)
- `threshold_force`: Threshold value used (N)
- `method`: "threshold"

**Algorithm**:
1. Apply optional smoothing to force signal
2. Calculate absolute force values
3. Find peak force and compute threshold
4. Search backward from peak to find start (force crosses threshold)
5. Search forward from peak to find end (force crosses threshold)
6. Calculate duration as end_time - start_time

#### `calculate_contact_duration_velocity(time, velocity, smooth=True) -> Dict`
Alternative method: Uses velocity reversal to detect contact.

**Parameters**:
- `time`: Time array (seconds)
- `velocity`: Velocity array (m/s)
- `smooth`: Apply smoothing (default: True)

**Returns**: Dictionary with contact metrics and `method`: "velocity"

**Algorithm**:
1. Smooth velocity signal if requested
2. Find velocity zero-crossings (direction reversals)
3. Identify contact period between first deceleration and rebound
4. Calculate duration

#### `calculate_contact_duration_energy(time, energy) -> Dict`
Alternative method: Uses energy changes to detect contact.

**Parameters**:
- `time`: Time array (seconds)
- `energy`: Kinetic energy array (J)

**Returns**: Dictionary with contact metrics and `method`: "energy"

**Algorithm**:
1. Identify maximum energy (before impact)
2. Find energy minimum (maximum compression)
3. Calculate contact duration from energy profile
4. Determine start/end based on energy thresholds

#### `save_annotated_profile(time, force, contact_info, thickness, run, output_dir)`
Creates and saves force-time plot with contact region highlighted.

**Parameters**:
- `time`: Time array
- `force`: Force array
- `contact_info`: Dictionary from contact duration calculation
- `thickness`: Pad thickness (mm)
- `run`: Run identifier
- `output_dir`: Directory for saving plot

**Output**: PNG file with:
- Full force-time profile
- Shaded contact region
- Vertical lines marking start/end of contact
- Annotated duration and peak force values

### Configuration

Modify these constants for custom analysis:

```python
# Smoothing parameters
WINDOW_LENGTH = 11  # Must be odd
POLYORDER = 3       # Polynomial order

# Threshold defaults
DEFAULT_THRESHOLD = 0.05  # 5% of peak force

# Plot settings
DPI = 300  # Resolution for saved figures
FIGSIZE = (12, 6)  # Figure dimensions in inches
```

## utils.py

### Purpose
Provides shared utility functions for statistical analysis, uncertainty calculations, and visualization.

### Key Functions

#### `format_uncertainty(value)`
Formats uncertainty values to appropriate significant figures.

**Parameters**:
- `value`: Numerical uncertainty value

**Returns**: String formatted to 1-2 significant figures, with scientific notation if needed

**Examples**:
```python
format_uncertainty(0.0234)  # Returns: "0.02"
format_uncertainty(0.000156)  # Returns: "1.6 \\times 10^{-4}"
format_uncertainty(0.123)  # Returns: "0.12"
```

**Rules**:
- 2 significant figures if first digit is 1
- 1 significant figure otherwise
- Scientific notation for values < 0.01
- Returns "N/A" for None, 0, or NaN

#### `perform_linear_regression_with_uncertainty(x, y, yerr)`
Performs linear regression with bootstrap uncertainty estimation.

**Parameters**:
- `x`: Independent variable array
- `y`: Dependent variable array
- `yerr`: Uncertainties in y values (optional)

**Returns**: Tuple of:
- `slope`: Regression slope
- `intercept`: Y-intercept
- `r_value`: Correlation coefficient
- `p_value`: Statistical significance
- `std_err`: Standard error of regression
- `slope_uncertainty`: Bootstrap uncertainty in slope
- `intercept_uncertainty`: Bootstrap uncertainty in intercept

**Algorithm**:
1. Perform standard linear regression using `scipy.stats.linregress`
2. Bootstrap resample data 1000 times
3. For each bootstrap sample:
   - Randomly sample with replacement
   - Add noise based on yerr if provided
   - Calculate regression parameters
4. Estimate uncertainties as standard deviation of bootstrap distribution

**Notes**:
- Returns None values if insufficient data (< 2 points)
- Handles missing yerr gracefully
- Uses fixed random seed for reproducibility in calling code

#### `calculate_slope_uncertainty_values(x, y, yerr)`
Calculates slope/intercept uncertainties from error bar extremes.

**Parameters**:
- `x`: Independent variable array
- `y`: Dependent variable array  
- `yerr`: Uncertainties in y values

**Returns**: Tuple of:
- `slope_uncertainty`: Half the difference between max and min slopes
- `intercept_uncertainty`: Half the difference between max and min intercepts

**Algorithm**:
1. Sort data by x values
2. Calculate minimum slope: connect (x1, y1+yerr1) to (x2, y2-yerr2)
3. Calculate maximum slope: connect (x1, y1-yerr1) to (x2, y2+yerr2)
4. Uncertainty = (max - min) / 2

**Use case**: Provides geometric estimate of uncertainty range visible in plots

#### `plot_slope_uncertainty_bounds(ax, x, y, yerr, x_fit, colors)`
Adds min/max slope lines to existing regression plot.

**Parameters**:
- `ax`: Matplotlib axis object
- `x`: Data x values
- `y`: Data y values
- `yerr`: Y uncertainties
- `x_fit`: X values for plotting fitted lines
- `colors`: Dictionary with color specifications

**Effect**: Adds two dotted lines showing extreme slopes based on error bars

#### `plot_regression_and_ci(ax, x, y, reg_stats, colors)`
Plots regression line with 95% confidence interval.

**Parameters**:
- `ax`: Matplotlib axis object
- `x`: Data x values
- `y`: Data y values
- `reg_stats`: Dictionary with regression results
- `colors`: Color specifications

**Features**:
- Best-fit line
- 95% confidence band using Student's t-distribution
- Extends slightly beyond data range for visibility

**Confidence interval calculation**:
```python
t_val = stats.t.ppf(0.975, n - 2)  # 95% CI
s_err = reg_stats['std_err']
x_mean = np.mean(x)
sxx = np.sum((x - x_mean) ** 2)
se_fit = s_err * np.sqrt(1/n + (x_fit - x_mean)**2 / sxx)
ci = t_val * se_fit
```

#### `plot_cv_subplot(ax, summary_df, colors, y_variable)`
Creates coefficient of variation plot.

**Parameters**:
- `ax`: Matplotlib axis object
- `summary_df`: Summary statistics DataFrame
- `colors`: Color dictionary
- `y_variable`: Column name for CV data

**Features**:
- Bar or scatter visualization
- Reference line for typical precision
- Automatic y-axis scaling based on data range

#### `sort_run_key(run_label: str) -> tuple`
Provides sorting key for run labels.

**Parameters**:
- `run_label`: Run identifier string (e.g., "Run 1", "Latest")

**Returns**: Tuple for proper sorting (numeric runs before "Latest")

**Example**:
```python
runs = ["Latest", "Run 2", "Run 1", "Run 10"]
sorted(runs, key=sort_run_key)
# Returns: ["Run 1", "Run 2", "Run 10", "Latest"]
```

### Shared Constants

```python
# Statistical parameters
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95

# Plotting defaults
DEFAULT_COLORS = {
    "gray": "#7f8c8d",
    "blue": "#3498db",
    "orange": "#e67e22",
    "pink": "#e74c3c"
}
```

## Dependencies

All scripts require:
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and CSV I/O
- **Matplotlib**: Visualization and plotting
- **SciPy**: Statistical functions and signal processing

Import statements:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter
```

## Error Handling

### Common Exceptions

1. **FileNotFoundError**: Input file doesn't exist
   - Check file path and ensure data files are in correct directory

2. **ValueError**: Invalid data format or parameters
   - Verify CSV structure matches expected format
   - Check that threshold values are between 0 and 1

3. **KeyError**: Missing required columns
   - Ensure aligned data has proper multi-level structure
   - Verify column names match expected format

### Validation Checks

Scripts perform automatic validation:
- Check for required columns in data files
- Verify numeric data types
- Filter out NaN and infinite values
- Ensure sufficient data points for analysis
- Validate parameter ranges

### Graceful Degradation

When issues occur:
- Individual failed runs are skipped with warnings
- Analysis continues with available valid data
- Missing optional columns trigger informative messages
- Results include quality metrics (n_runs, CV) to assess reliability
