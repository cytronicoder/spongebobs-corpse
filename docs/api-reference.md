# API Reference

## Module: analysis.py

### Functions

#### plot_duration_subplot

```python
def plot_duration_subplot(ax, detailed_df, summary_df, colors)
```

Create scatter plot of contact duration vs. thickness with regression analysis.

**Parameters:**

- `ax` (matplotlib.axes.Axes): The axes object to plot on
- `detailed_df` (pd.DataFrame): DataFrame containing individual run measurements with columns:
  - `thickness_mm`: Pad thickness in millimeters
  - `duration_s`: Contact duration in seconds
  - `run`: Run identifier
- `summary_df` (pd.DataFrame): DataFrame containing summary statistics with columns:
  - `thickness_mm`: Pad thickness
  - `duration_s_mean`: Mean contact duration
  - `duration_s_std`: Standard deviation
- `colors` (dict): Dictionary mapping color names to hex codes

**Returns:**
None (modifies `ax` in place)

**Side Effects:**

- Adds scatter points, error bars, regression line, and confidence intervals to the axes
- Sets labels, title, legend, and grid

**Example:**

```python
fig, ax = plt.subplots()
colors = {"gray": "#7f8c8d", "blue": "#3498db", "orange": "#e67e22", "pink": "#e74c3c"}
plot_duration_subplot(ax, detailed_df, summary_df, colors)
plt.show()
```

#### plot_peak_force_subplot

```python
def plot_peak_force_subplot(ax, detailed_df, summary_df, colors)
```

Create scatter plot of peak force vs. thickness with regression analysis.

**Parameters:**

- `ax` (matplotlib.axes.Axes): The axes object to plot on
- `detailed_df` (pd.DataFrame): DataFrame with columns:
  - `thickness_mm`: Pad thickness
  - `peak_force_N`: Peak force in Newtons
  - `run`: Run identifier
- `summary_df` (pd.DataFrame): Summary statistics DataFrame
- `colors` (dict): Color specifications

**Returns:**
None (modifies `ax` in place)

#### plot_cv_subplot

```python
def plot_cv_subplot(ax, summary_df, colors, y_variable)
```

Plot coefficient of variation vs. thickness.

**Parameters:**

- `ax` (matplotlib.axes.Axes): The axes object
- `summary_df` (pd.DataFrame): Summary statistics
- `colors` (dict): Color dictionary
- `y_variable` (str): Column name for CV data (e.g., `"duration_s_cv"` or `"peak_force_N_cv"`)

**Returns:**
None (modifies `ax` in place)

**Notes:**

- Automatically scales y-axis based on maximum CV value
- Adds reference line at typical precision threshold

#### create_regression_text

```python
def create_regression_text(slope, intercept, r_value, p_value,
                          slope_unc, intercept_unc, var_name)
```

Generate formatted text annotation for regression statistics.

**Parameters:**

- `slope` (float): Regression slope coefficient
- `intercept` (float): Y-intercept value
- `r_value` (float): Correlation coefficient (-1 to 1)
- `p_value` (float): Statistical significance (0 to 1)
- `slope_unc` (float): Uncertainty in slope from bootstrap
- `intercept_unc` (float): Uncertainty in intercept from bootstrap
- `var_name` (str): Variable name for equation (e.g., "duration", "force")

**Returns:**

- `str`: Formatted text with LaTeX-style equations and statistics

**Example Output:**

```
y = (1.23 ± 0.04) × 10⁻⁴ x + (0.0198 ± 0.0015)
R² = 0.987, p = 0.0023
```

#### generate_statistical_report

```python
def generate_statistical_report(detailed_df, summary_df, output_dir)
```

Create comprehensive text report of statistical analysis.

**Parameters:**

- `detailed_df` (pd.DataFrame): Individual run measurements
- `summary_df` (pd.DataFrame): Summary statistics
- `output_dir` (str or Path): Directory path for output file

**Returns:**
None

**Side Effects:**

- Creates `statistical_report.txt` in `output_dir`
- File contains regression results, CV analysis, and significance tests

**Raises:**

- `IOError`: If unable to write to output directory

## Module: contact_duration.py

### Functions

#### extract_thickness_from_filename

```python
def extract_thickness_from_filename(filename: str) -> Optional[float]
```

Extract pad thickness from filename using regex pattern.

**Parameters:**

- `filename` (str): Filename or path containing thickness information

**Returns:**

- `float`: Thickness in millimeters, or `None` if pattern not found

**Pattern Matched:**

- `(\d+(?:\.\d+)?)\s*mm` (case-insensitive)

**Examples:**

```python
extract_thickness_from_filename("experiment - 40mm.csv")  # 40.0
extract_thickness_from_filename("test_4.5mm_data.csv")   # 4.5
extract_thickness_from_filename("no_thickness.csv")       # None
```

#### detect_runs

```python
def detect_runs(columns: pd.MultiIndex) -> List[str]
```

Extract unique run labels from DataFrame multi-level column headers.

**Parameters:**

- `columns` (pd.MultiIndex): Multi-level column index from aligned DataFrame

**Returns:**

- `List[str]`: Sorted list of run identifiers (e.g., `["Run 1", "Run 2", "Latest"]`)

**Notes:**

- Sorts numerically for "Run X" labels
- Places "Latest" at the end

#### smooth_signal

```python
def smooth_signal(data: np.ndarray, window_length: int = 11,
                 polyorder: int = 3) -> np.ndarray
```

Apply Savitzky-Golay filter to smooth noisy signals.

**Parameters:**

- `data` (np.ndarray): Input signal array
- `window_length` (int, optional): Filter window size (must be odd). Default: 11
- `polyorder` (int, optional): Polynomial order for fitting. Default: 3

**Returns:**

- `np.ndarray`: Smoothed signal (same shape as input)

**Notes:**

- Returns original data if `len(data) < window_length`
- Handles NaN values by masking
- Uses `scipy.signal.savgol_filter` internally

**Raises:**

- `ValueError`: If `window_length` is even or `polyorder >= window_length`

#### calculate_contact_duration_threshold

```python
def calculate_contact_duration_threshold(
    time: np.ndarray,
    force: np.ndarray,
    threshold_fraction: float = 0.05,
    smooth: bool = True
) -> Dict[str, float]
```

Calculate contact duration using force threshold method.

**Parameters:**

- `time` (np.ndarray): Time array in seconds (must be aligned)
- `force` (np.ndarray): Force array in Newtons
- `threshold_fraction` (float, optional): Fraction of peak force for threshold. Default: 0.05
- `smooth` (bool, optional): Apply smoothing filter. Default: True

**Returns:**

- `dict`: Dictionary with keys:
  - `duration` (float): Contact duration in seconds
  - `start_time` (float): Time when contact starts (s)
  - `end_time` (float): Time when contact ends (s)
  - `peak_force` (float): Maximum absolute force (N)
  - `threshold_force` (float): Threshold value used (N)
  - `method` (str): "threshold"

**Algorithm:**

1. Apply optional smoothing to force signal
2. Calculate absolute force and find peak
3. Compute threshold = `threshold_fraction * peak_force`
4. Search backward from peak for crossing threshold (start)
5. Search forward from peak for crossing threshold (end)
6. Return `duration = end_time - start_time`

**Notes:**

- Returns NaN values if insufficient valid data
- Filters out NaN and Inf values automatically

**Example:**

```python
contact_info = calculate_contact_duration_threshold(
    time=time_array,
    force=force_array,
    threshold_fraction=0.05,
    smooth=True
)
print(f"Duration: {contact_info['duration']:.4f} s")
print(f"Peak force: {contact_info['peak_force']:.2f} N")
```

#### calculate_contact_duration_velocity

```python
def calculate_contact_duration_velocity(
    time: np.ndarray,
    velocity: np.ndarray,
    smooth: bool = True,
    velocity_threshold: float = 0.1
) -> Dict[str, float]
```

Calculate contact duration using velocity threshold method.

**Parameters:**

- `time` (np.ndarray): Time array in seconds
- `velocity` (np.ndarray): Velocity array in m/s
- `smooth` (bool, optional): Apply smoothing. Default: True
- `velocity_threshold` (float, optional): Velocity threshold for contact detection in m/s. Default: 0.1

**Returns:**

- `dict`: Dictionary with contact metrics:
  - `duration` (float): Contact duration in seconds
  - `start_time` (float): Contact start time
  - `end_time` (float): Contact end time
  - `min_velocity` (float): Minimum velocity (maximum compression)
  - `velocity_threshold` (float): Threshold value used
  - `method` (str): "velocity"

**Algorithm:**

1. Filter out NaN values from sparse velocity data
2. Validate minimum 10 valid measurements
3. Optionally smooth velocity signal (Savitzky-Golay)
4. Find minimum velocity (maximum compression point)
5. Search backward for contact start (v > threshold)
6. Search forward for contact end (|v| > threshold)
7. Calculate duration from threshold crossings

**Notes:**

- Handles sparse velocity data with NaN values robustly
- Requires at least 10 valid velocity measurements
- Threshold-based detection more stable than zero-crossing
- Useful for validation and comparison with force method

#### calculate_contact_duration_energy

```python
def calculate_contact_duration_energy(
    time: np.ndarray,
    velocity: np.ndarray,
    mass: float = 0.5,
    recovery_fraction: float = 0.5,
    smooth: bool = True
) -> Dict[str, float]
```

Calculate contact duration using kinetic energy tracking method.

**Parameters:**

- `time` (np.ndarray): Time array in seconds
- `velocity` (np.ndarray): Velocity array in m/s
- `mass` (float, optional): Cart mass in kg. Default: 0.5
- `recovery_fraction` (float, optional): Fraction of initial energy for contact boundaries. Default: 0.5
- `smooth` (bool, optional): Apply smoothing. Default: True

**Returns:**

- `dict`: Dictionary with contact metrics:
  - `duration` (float): Contact duration in seconds
  - `start_time` (float): Contact start time
  - `end_time` (float): Contact end time
  - `initial_energy` (float): Maximum kinetic energy before impact (J)
  - `min_energy` (float): Minimum kinetic energy during compression (J)
  - `recovery_threshold` (float): Energy threshold used (J)
  - `method` (str): "energy"

**Algorithm:**

1. Filter out NaN values from velocity data
2. Validate minimum 10 valid measurements
3. Calculate kinetic energy: E_k = 0.5 * m * v²
4. Find global maximum energy (initial velocity before impact)
5. Find minimum energy AFTER maximum (compression during impact)
6. Calculate recovery threshold (fraction of initial energy)
7. Search backward from minimum for contact start (E < threshold)
8. Search forward from minimum for contact end (E ≥ threshold)
9. Calculate duration

**Notes:**

- Requires velocity data and cart mass parameter
- Finds correct impact event (not spurious low-energy regions)
- Global maximum search ensures proper event identification
- Recovery fraction affects contact boundary detection
- Requires at least 10 valid velocity measurements

#### save_annotated_profile

```python
def save_annotated_profile(time, force, contact_info, thickness, run, output_dir)
```

Create and save force-time plot with contact region highlighted.

**Parameters:**

- `time` (np.ndarray): Time array
- `force` (np.ndarray): Force array
- `contact_info` (dict): Dictionary from contact duration calculation
- `thickness` (float): Pad thickness in mm
- `run` (str): Run identifier
- `output_dir` (str or Path): Directory for saving plot

**Returns:**
None

**Side Effects:**

- Creates PNG file: `{output_dir}/{thickness}mm_{run}.png`
- Creates `output_dir/annotated_profiles/` if it doesn't exist

**Plot Features:**

- Full force-time profile
- Shaded contact region (green/yellow)
- Vertical lines marking contact start/end
- Annotated duration and peak force values
- Grid and proper labeling

**Raises:**

- `IOError`: If unable to create directory or save file

## Module: utils.py

### Functions

#### format_uncertainty

```python
def format_uncertainty(value)
```

Format uncertainty to appropriate significant figures.

**Parameters:**

- `value` (float or None): Uncertainty value to format

**Returns:**

- `str`: Formatted string with 1-2 significant figures

**Formatting Rules:**

- Returns "N/A" for None, 0, or NaN
- 2 significant figures if first digit is 1
- 1 significant figure otherwise
- Scientific notation for values < 0.01 (LaTeX format)

**Examples:**

```python
format_uncertainty(0.0234)     # "0.02"
format_uncertainty(0.123)      # "0.12"  (first digit is 1)
format_uncertainty(0.000156)   # "1.6 \\times 10^{-4}"
format_uncertainty(None)       # "N/A"
```

#### perform_linear_regression_with_uncertainty

```python
def perform_linear_regression_with_uncertainty(x, y, yerr)
```

Perform linear regression with bootstrap uncertainty estimation.

**Parameters:**

- `x` (np.ndarray): Independent variable array
- `y` (np.ndarray): Dependent variable array
- `yerr` (np.ndarray or None): Uncertainties in y values (optional)

**Returns:**

- `tuple`: Seven values:
  1. `slope` (float): Regression slope
  2. `intercept` (float): Y-intercept
  3. `r_value` (float): Correlation coefficient
  4. `p_value` (float): Statistical significance
  5. `std_err` (float): Standard error of regression
  6. `slope_uncertainty` (float): Bootstrap uncertainty in slope
  7. `intercept_uncertainty` (float): Bootstrap uncertainty in intercept

**Algorithm:**

1. Perform standard linear regression using `scipy.stats.linregress`
2. Bootstrap resample data 1000 times:
   - Sample with replacement
   - Add noise based on yerr if provided
   - Calculate regression parameters
3. Estimate uncertainties as standard deviation of bootstrap distribution

**Notes:**

- Returns `(None, None, None, None, None, None, None)` if `len(x) < 2`
- Handles missing yerr gracefully
- Bootstrap provides robust uncertainty estimates

**Example:**

```python
slope, intercept, r, p, stderr, slope_unc, int_unc = \
    perform_linear_regression_with_uncertainty(x, y, yerr)

print(f"Slope: {slope:.4f} ± {slope_unc:.4f}")
print(f"R² = {r**2:.3f}, p = {p:.4f}")
```

#### calculate_slope_uncertainty_values

```python
def calculate_slope_uncertainty_values(x, y, yerr)
```

Calculate slope/intercept uncertainties from error bar extremes.

**Parameters:**

- `x` (np.ndarray): Independent variable
- `y` (np.ndarray): Dependent variable
- `yerr` (np.ndarray): Y uncertainties

**Returns:**

- `tuple`: Two values:
  1. `slope_uncertainty` (float): Half-range of slopes
  2. `intercept_uncertainty` (float): Half-range of intercepts

**Algorithm:**

1. Sort data by x values
2. Calculate minimum slope: `(y[0]+yerr[0], x[0])` to `(y[-1]-yerr[-1], x[-1])`
3. Calculate maximum slope: `(y[0]-yerr[0], x[0])` to `(y[-1]+yerr[-1], x[-1])`
4. Uncertainty = `(max - min) / 2`

**Notes:**

- Returns `(None, None)` if insufficient data or missing yerr
- Provides geometric estimate of uncertainty visible in plots
- Complements bootstrap uncertainty estimation

#### plot_slope_uncertainty_bounds

```python
def plot_slope_uncertainty_bounds(ax, x, y, yerr, x_fit, colors)
```

Add min/max slope lines to existing regression plot.

**Parameters:**

- `ax` (matplotlib.axes.Axes): Axes object
- `x` (np.ndarray): Data x values
- `y` (np.ndarray): Data y values
- `yerr` (np.ndarray): Y uncertainties
- `x_fit` (np.ndarray): X values for plotting fitted lines
- `colors` (dict): Color specifications (requires `colors["pink"]`)

**Returns:**
None (modifies `ax` in place)

**Side Effects:**

- Adds two dotted lines showing extreme slopes
- Lines use pink color with 80% opacity
- Labeled "Min slope" and "Max slope"

#### plot_regression_and_ci

```python
def plot_regression_and_ci(ax, x, y, reg_stats, colors)
```

Plot regression line with 95% confidence interval.

**Parameters:**

- `ax` (matplotlib.axes.Axes): Axes object
- `x` (np.ndarray): Data x values
- `y` (np.ndarray): Data y values
- `reg_stats` (dict): Dictionary with regression results:
  - `slope`: Regression slope
  - `intercept`: Y-intercept
  - `std_err`: Standard error
- `colors` (dict): Color specifications (requires `colors["orange"]`)

**Returns:**
None (modifies `ax` in place)

**Features:**

- Best-fit line (solid orange)
- 95% confidence band (shaded orange, 20% opacity)
- Extends 5% beyond data range for visibility

**Confidence Interval Formula:**

```
CI = t_0.975(n-2) * SE * sqrt(1/n + (x_fit - x_mean)² / Sxx)
```

where:

- `n` = number of points
- `SE` = standard error
- `Sxx` = sum of squared deviations in x

#### plot_cv_subplot

```python
def plot_cv_subplot(ax, summary_df, colors, y_variable)
```

Create coefficient of variation subplot.

**Parameters:**

- `ax` (matplotlib.axes.Axes): Axes object
- `summary_df` (pd.DataFrame): Summary statistics DataFrame
- `colors` (dict): Color dictionary
- `y_variable` (str): Column name for CV data

**Returns:**
None (modifies `ax` in place)

**Features:**

- Scatter or bar visualization of CV values
- Reference line for typical precision threshold (e.g., 5%)
- Automatic y-axis scaling

**Notes:**

- CV = (standard deviation / mean) × 100%
- Lower CV indicates better precision

#### sort_run_key

```python
def sort_run_key(run_label: str) -> tuple
```

Provide sorting key for run labels.

**Parameters:**

- `run_label` (str): Run identifier (e.g., "Run 1", "Latest")

**Returns:**

- `tuple`: Sort key (numeric runs before non-numeric)

**Behavior:**

- "Run X" labels: extracted as integers for numeric sorting
- Other labels (e.g., "Latest"): sorted alphabetically after numeric runs

**Example:**

```python
runs = ["Latest", "Run 10", "Run 2", "Run 1"]
sorted(runs, key=sort_run_key)
# Result: ["Run 1", "Run 2", "Run 10", "Latest"]
```

## Data Types

### DataFrame Schemas

#### detailed_df (Individual Measurements)

```python
{
    "filename": str,
    "thickness_mm": float,
    "run": str,
    "duration_s": float,
    "peak_force_N": float,
    "start_time_s": float,
    "end_time_s": float,
    "threshold_force_N": float,
    "method": str
}
```

#### summary_df (Summary Statistics)

```python
{
    "filename": str,
    "thickness_mm": float,
    "n_runs": int,
    "duration_s_mean": float,
    "duration_s_std": float,
    "duration_s_cv": float,
    "peak_force_N_mean": float,
    "peak_force_N_std": float,
    "peak_force_N_cv": float
}
```

#### contact_info (Contact Duration Result)

```python
{
    "duration": float,        # seconds
    "start_time": float,      # seconds
    "end_time": float,        # seconds
    "peak_force": float,      # Newtons
    "threshold_force": float, # Newtons (threshold method only)
    "method": str            # "threshold", "velocity", or "energy"
}
```

## Constants

### Bootstrap Parameters

```python
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95
```

### Default Colors

```python
DEFAULT_COLORS = {
    "gray": "#7f8c8d",    # Individual points
    "blue": "#3498db",    # Mean values
    "orange": "#e67e22",  # Regression line
    "pink": "#e74c3c"     # Uncertainty bounds
}
```

### Smoothing Parameters

```python
WINDOW_LENGTH = 11  # Savitzky-Golay filter window
POLYORDER = 3       # Polynomial order
```

### Threshold Defaults

```python
DEFAULT_THRESHOLD = 0.05  # 5% of peak force
```

## Exception Handling

### Common Exceptions

- `ValueError`: Invalid parameters or data format
- `KeyError`: Missing required columns in DataFrame
- `IOError`: File reading/writing errors
- `TypeError`: Incorrect data types

### Error Recovery

Functions generally handle errors gracefully:

- Return None or empty results for invalid inputs
- Skip problematic data points with warnings
- Continue processing valid data when possible
- Provide informative error messages
