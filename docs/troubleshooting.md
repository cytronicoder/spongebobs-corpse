# Troubleshooting

## Common Issues and Solutions

### Installation Issues

#### Problem: `ModuleNotFoundError: No module named 'numpy'`

**Cause**: Required package not installed.

**Solution**:

```bash
pip install numpy pandas matplotlib scipy
```

**If using virtual environment**:

```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install packages
pip install numpy pandas matplotlib scipy
```

#### Problem: `ImportError: cannot import name 'savgol_filter'`

**Cause**: Outdated scipy version.

**Solution**:

```bash
pip install --upgrade scipy
```

#### Problem: Permission denied when installing packages

**Cause**: Insufficient permissions for global installation.

**Solution**:

```bash
# Option 1: Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install numpy pandas matplotlib scipy

# Option 2: Install for user only
pip install --user numpy pandas matplotlib scipy
```

### Data Loading Issues

#### Problem: `FileNotFoundError: No such file or directory`

**Cause**: Incorrect file path or file doesn't exist.

**Solution**:

```bash
# Verify file exists
ls -lh "outputs/dr lee go brr - 4mm/dr lee go brr - 4mm_aligned.csv"

# Use absolute path
python contact_duration.py "/full/path/to/file.csv"

# Check current directory
pwd
```

#### Problem: `pd.errors.ParserError: Error tokenizing data`

**Cause**: Malformed CSV file or incorrect encoding.

**Solution**:

1. Check CSV structure:

   ```bash
   head -20 file.csv
   ```

2. Verify encoding:

   ```bash
   file -I file.csv  # Check encoding
   ```

3. Convert if needed:
   ```bash
   iconv -f ISO-8859-1 -t UTF-8 file.csv > file_utf8.csv
   ```

#### Problem: `KeyError: 'force_N'` or missing column error

**Cause**: Column names don't match expected format.

**Solution**:

1. Check column names:

   ```python
   import pandas as pd
   df = pd.read_csv("file.csv", header=[0,1])
   print(df.columns)
   ```

2. Verify multi-level structure:

   - Level 0: Run identifiers (Run 1, Run 2, etc.)
   - Level 1: Data types (time_s, force_N)

3. Rename columns if needed:
   ```python
   # Example fix
   df.columns = pd.MultiIndex.from_tuples([
       ('Run 1', 'time_s'), ('Run 1', 'force_N'),
       ('Run 2', 'time_s'), ('Run 2', 'force_N')
   ])
   ```

### Analysis Issues

#### Problem: `ValueError: x and y must have same first dimension`

**Cause**: Mismatched array lengths in regression.

**Solution**:

1. Check data filtering:

   ```python
   print(f"x length: {len(x)}, y length: {len(y)}")
   ```

2. Ensure complete data:
   ```python
   # Remove NaN values consistently
   mask = ~(np.isnan(x) | np.isnan(y))
   x_clean = x[mask]
   y_clean = y[mask]
   ```

#### Problem: Contact duration is negative or zero

**Cause**: Incorrect threshold or misaligned data.

**Solution**:

1. Check force signal:

   ```python
   import matplotlib.pyplot as plt
   plt.plot(time, force)
   plt.xlabel('Time (s)')
   plt.ylabel('Force (N)')
   plt.show()
   ```

2. Adjust threshold:

   ```bash
   # Try different thresholds
   python contact_duration.py file.csv --threshold 0.03  # Lower
   python contact_duration.py file.csv --threshold 0.10  # Higher
   ```

3. Verify alignment:
   - Peaks should be synchronized across runs
   - Check alignment preprocessing

#### Problem: `RuntimeWarning: invalid value encountered in divide`

**Cause**: Division by zero or NaN values.

**Solution**:

```python
# Check for zero/NaN values
print(f"Zero values in denominator: {np.sum(denominator == 0)}")
print(f"NaN values: {np.sum(np.isnan(denominator))}")

# Add safeguards
denominator_safe = np.where(denominator == 0, np.nan, denominator)
result = numerator / denominator_safe
```

#### Problem: All runs show identical contact durations

**Cause**: Data not properly separated by run, or smoothing over-applied.

**Solution**:

1. Verify run separation:

   ```python
   runs = detect_runs(df.columns)
   print(f"Detected runs: {runs}")
   ```

2. Check smoothing parameters:

   ```bash
   # Disable smoothing
   python contact_duration.py file.csv --no-smooth
   ```

3. Reduce smoothing window:
   ```python
   # In contact_duration.py
   window_length = 5  # Reduced from 11
   ```

### Visualization Issues

#### Problem: Plot shows no data or empty figure

**Cause**: Data filtering removed all points, or display issue.

**Solution**:

```python
# Check data before plotting
print(f"Data points: {len(x)}")
print(f"X range: {x.min()} to {x.max()}")
print(f"Y range: {y.min()} to {y.max()}")

# Explicitly show plot
plt.show()

# Or save instead
plt.savefig('debug_plot.png')
```

#### Problem: Regression line not visible

**Cause**: Extreme axis limits or line color issue.

**Solution**:

```python
# Check fitted values
print(f"y_fit range: {y_fit.min()} to {y_fit.max()}")

# Adjust axis limits
ax.set_xlim(x.min() * 0.9, x.max() * 1.1)
ax.set_ylim(y.min() * 0.9, y.max() * 1.1)

# Use contrasting color
ax.plot(x_fit, y_fit, 'r-', linewidth=3)  # Red, thick line
```

#### Problem: Annotated profiles show wrong contact region

**Cause**: Threshold too high/low or baseline offset.

**Solution**:

1. Visualize threshold:

   ```python
   plt.plot(time, force, label='Force')
   plt.axhline(y=threshold_force, color='r', linestyle='--', label='Threshold')
   plt.axvline(x=start_time, color='g', linestyle='--', label='Start')
   plt.axvline(x=end_time, color='b', linestyle='--', label='End')
   plt.legend()
   plt.show()
   ```

2. Adjust threshold:

   - Too early start: Increase threshold
   - Too late start: Decrease threshold

3. Check baseline:
   ```python
   # Remove baseline offset
   baseline = np.median(force[:100])  # Use first 100 points
   force_corrected = force - baseline
   ```

### Statistical Issues

#### Problem: `p-value = nan` or regression fails

**Cause**: Insufficient data points or perfect correlation.

**Solution**:

```python
# Check data points
print(f"Number of points: {len(x)}")

# Need at least 3 points for meaningful regression
if len(x) < 3:
    print("Insufficient data for regression")
```

#### Problem: R² is negative

**Cause**: Model fits worse than horizontal line (mean).

**Solution**:

1. Check data quality:

   ```python
   plt.scatter(x, y)
   plt.xlabel('Thickness')
   plt.ylabel('Duration')
   plt.show()
   ```

2. Verify expected relationship:

   - Duration should increase with thickness
   - Force should decrease with thickness

3. Investigate outliers:

   ```python
   # Calculate residuals
   y_pred = slope * x + intercept
   residuals = y - y_pred

   # Identify large residuals
   outliers = np.abs(residuals) > 2 * np.std(residuals)
   print(f"Outliers: {np.sum(outliers)}")
   ```

#### Problem: Very large uncertainties in regression parameters

**Cause**: High data scatter or small sample size.

**Solution**:

1. Increase number of runs per thickness
2. Improve experimental control (reduce CV)
3. Check for systematic errors
4. Consider weighted regression if uncertainties vary

### Output Issues

#### Problem: No output files generated

**Cause**: Script error, permissions, or incorrect output path.

**Solution**:

```bash
# Check for errors in terminal output
python contact_duration.py file.csv 2>&1 | tee output.log

# Verify output directory exists
mkdir -p outputs/contact_analysis_5pct

# Check write permissions
ls -ld outputs/contact_analysis_5pct
chmod 755 outputs/contact_analysis_5pct
```

#### Problem: CSV files contain only headers

**Cause**: No valid data processed or all runs failed.

**Solution**:

1. Check console output for warnings
2. Verify input data validity:

   ```python
   df = pd.read_csv("aligned.csv", header=[0,1])
   print(f"Runs detected: {detect_runs(df.columns)}")
   print(f"Valid force values: {np.isfinite(df['Run 1']['force_N']).sum()}")
   ```

3. Lower quality thresholds:
   - Allow more missing data
   - Relax validation criteria

#### Problem: Plots saved but appear blank

**Cause**: Data plotted outside visible range or transparent.

**Solution**:

```python
# Before saving, check axis limits
ax.relim()
ax.autoscale_view()

# Increase figure DPI
plt.savefig('plot.png', dpi=300, bbox_inches='tight')

# Use white background
fig.patch.set_facecolor('white')
```

### Performance Issues

#### Problem: Script runs very slowly

**Cause**: Large dataset or inefficient operations.

**Solution**:

1. Reduce bootstrap iterations:

   ```python
   n_boot = 100  # Reduced from 1000
   ```

2. Optimize smoothing:

   ```python
   # Skip smoothing for large datasets
   smooth = False
   ```

3. Profile code:
   ```bash
   python -m cProfile -o profile.stats contact_duration.py file.csv
   ```

#### Problem: Memory error with large files

**Cause**: Insufficient RAM for dataset.

**Solution**:

1. Process in chunks:

   ```python
   # Read CSV in chunks
   chunk_size = 10000
   for chunk in pd.read_csv(file, chunksize=chunk_size):
       process(chunk)
   ```

2. Reduce data precision:

   ```python
   df = pd.read_csv(file, dtype={'force_N': 'float32'})
   ```

3. Delete unused data:
   ```python
   del large_array
   import gc
   gc.collect()
   ```

## Error Messages Reference

### Common Errors

| Error Message                     | Likely Cause              | Solution                            |
| --------------------------------- | ------------------------- | ----------------------------------- |
| `FileNotFoundError`               | File path incorrect       | Check path, use absolute path       |
| `KeyError: 'column_name'`         | Missing column            | Verify CSV structure                |
| `ValueError: array size mismatch` | Inconsistent data lengths | Check for NaN, filter consistently  |
| `RuntimeWarning: divide by zero`  | Zero in denominator       | Add safeguards, check data          |
| `MemoryError`                     | Dataset too large         | Process in chunks, reduce precision |
| `ImportError`                     | Missing package           | Install required packages           |
| `TypeError: unsupported operand`  | Wrong data type           | Convert to numeric, check types     |
| `IndexError: index out of range`  | Array access error        | Check array lengths, use bounds     |

### Debug Mode

Add verbose output for troubleshooting:

```python
# In scripts, add debug flag
import sys

DEBUG = True  # Set to True for debugging

if DEBUG:
    print(f"Processing file: {filename}")
    print(f"Detected thickness: {thickness}")
    print(f"Number of runs: {len(runs)}")
    print(f"Force range: {force.min():.2f} to {force.max():.2f} N")
```

### Logging

Set up logging for persistent debugging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.debug("Starting analysis...")
```

## Getting Help

### Before Asking for Help

1. **Check error message**: Read full error traceback
2. **Review documentation**: Consult relevant doc sections
3. **Verify data**: Ensure input files are correct format
4. **Test with sample**: Try with provided example data
5. **Search issues**: Check if problem previously reported

### Reporting Issues

Include:

1. **Error message**: Full traceback
2. **Environment**: Python version, OS, package versions
3. **Data sample**: Minimal example that reproduces issue
4. **Expected behavior**: What should happen
5. **Actual behavior**: What actually happens
6. **Steps to reproduce**: Exact commands run

### Contact

For project-specific questions:

- Repository: github.com/cytronicoder/spongebobs-corpse
- Issues: Create GitHub issue with details above

For general Python/data analysis questions:

- Stack Overflow: Tag with `python`, `pandas`, `scipy`
- Documentation: numpy.org, pandas.pydata.org, scipy.org
