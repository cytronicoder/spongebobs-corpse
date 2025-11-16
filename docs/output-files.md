# Output Files

## Directory Structure

All analysis outputs are organized in the `outputs/` directory:

```
outputs/
├── contact_analysis_<threshold>pct/
│   ├── contact_duration_detailed.csv
│   ├── contact_duration_summary.csv
│   ├── statistical_report.txt
│   ├── summary_table_formatted.csv
│   └── annotated_profiles/
│       ├── 4mm_Run_1.png
│       ├── 4mm_Run_2.png
│       └── ...
└── dr lee go brr - <thickness>mm/
    ├── dr lee go brr - <thickness>mm_aligned.csv
    └── ...
```

## Detailed Results

### File: `contact_duration_detailed.csv`

**Purpose**: Individual measurements for each experimental run.

**Location**: `outputs/contact_analysis_<threshold>pct/`

**Format**: CSV with headers

**Columns**:
| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `filename` | str | - | Source aligned data file |
| `thickness_mm` | float | mm | Pad thickness |
| `run` | str | - | Run identifier (e.g., "Run 1", "Run 2") |
| `duration_s` | float | s | Contact duration |
| `peak_force_N` | float | N | Maximum absolute force during contact |
| `start_time_s` | float | s | Time when contact begins |
| `end_time_s` | float | s | Time when contact ends |
| `threshold_force_N` | float | N | Threshold value used for detection |
| `method` | str | - | Detection method ("threshold", "velocity", "energy") |

**Example Data**:
```csv
filename,thickness_mm,run,duration_s,peak_force_N,start_time_s,end_time_s,threshold_force_N,method
dr lee go brr - 4mm_aligned.csv,4.0,Run 1,0.0234,45.2,0.098,0.1214,2.26,threshold
dr lee go brr - 4mm_aligned.csv,4.0,Run 2,0.0228,44.8,0.099,0.1218,2.24,threshold
dr lee go brr - 4mm_aligned.csv,4.0,Run 3,0.0231,45.0,0.098,0.1211,2.25,threshold
dr lee go brr - 40mm_aligned.csv,40.0,Run 1,0.0456,28.3,0.085,0.1306,1.42,threshold
...
```

**Usage**:
- Statistical analysis of individual runs
- Outlier detection
- Run-to-run variability assessment
- Detailed data export for further processing

## Summary Statistics

### File: `contact_duration_summary.csv`

**Purpose**: Aggregated statistics for each thickness.

**Location**: `outputs/contact_analysis_<threshold>pct/`

**Format**: CSV with headers

**Columns**:
| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `filename` | str | - | Source aligned data file |
| `thickness_mm` | float | mm | Pad thickness |
| `n_runs` | int | - | Number of successful runs analyzed |
| `duration_s_mean` | float | s | Mean contact duration |
| `duration_s_std` | float | s | Standard deviation of duration |
| `duration_s_cv` | float | % | Coefficient of variation for duration |
| `peak_force_N_mean` | float | N | Mean peak force |
| `peak_force_N_std` | float | N | Standard deviation of peak force |
| `peak_force_N_cv` | float | % | Coefficient of variation for peak force |

**Example Data**:
```csv
filename,thickness_mm,n_runs,duration_s_mean,duration_s_std,duration_s_cv,peak_force_N_mean,peak_force_N_std,peak_force_N_cv
dr lee go brr - 4mm_aligned.csv,4.0,5,0.0231,0.0003,1.30,45.0,0.5,1.11
dr lee go brr - 40mm_aligned.csv,40.0,5,0.0456,0.0012,2.63,28.3,0.8,2.83
dr lee go brr - 52mm_aligned.csv,52.0,5,0.0589,0.0018,3.06,22.1,0.9,4.07
```

**Usage**:
- Regression analysis (thickness vs. duration/force)
- Precision assessment via CV
- Comparison across thicknesses
- Publication-ready data tables

## Formatted Summary

### File: `summary_table_formatted.csv`

**Purpose**: LaTeX-compatible formatted table for publication.

**Location**: `outputs/contact_analysis_<threshold>pct/`

**Format**: CSV with LaTeX formatting

**Columns**:
| Column | Description |
|--------|-------------|
| `Thickness (mm)` | Pad thickness |
| `Duration (s)` | Mean ± standard deviation |
| `CV (%)` | Coefficient of variation for duration |
| `Peak Force (N)` | Mean ± standard deviation |
| `CV (%)` | Coefficient of variation for force |

**Example Data**:
```csv
Thickness (mm),Duration (s),CV (%),Peak Force (N),CV (%)
4,$0.0231 \pm 0.0003$,1.30,$45.0 \pm 0.5$,1.11
40,$0.0456 \pm 0.0012$,2.63,$28.3 \pm 0.8$,2.83
52,$0.0589 \pm 0.0018$,3.06,$22.1 \pm 0.9$,4.07
```

**Usage**:
- Direct import into LaTeX documents
- Copy-paste into research papers
- Professional presentation of results

**LaTeX Example**:
```latex
\begin{table}[h]
\centering
\caption{Contact Duration and Peak Force vs. Pad Thickness}
\input{summary_table_formatted.csv}  % Or manually format
\end{table}
```

## Statistical Report

### File: `statistical_report.txt`

**Purpose**: Comprehensive text report of statistical analysis.

**Location**: `outputs/contact_analysis_<threshold>pct/`

**Format**: Plain text with sections

**Sections**:

#### 1. Header
```
Contact Duration Analysis Statistical Report
============================================
Analysis Date: 2025-11-16
Threshold: 5%
```

#### 2. Linear Regression: Duration vs Thickness
```
Linear Regression: Duration vs Thickness
-----------------------------------------
Slope: (1.23 ± 0.04) × 10⁻⁴ s/mm
Intercept: (0.0198 ± 0.0015) s
R² = 0.987
p-value = 0.0023

Regression Equation:
duration = (1.23 ± 0.04) × 10⁻⁴ × thickness + (0.0198 ± 0.0015)

The relationship is statistically significant (p < 0.05).
```

#### 3. Linear Regression: Peak Force vs Thickness
```
Linear Regression: Peak Force vs Thickness
-------------------------------------------
Slope: (-0.48 ± 0.03) N/mm
Intercept: (47.2 ± 1.2) N
R² = 0.994
p-value = 0.0008

Regression Equation:
force = (-0.48 ± 0.03) × thickness + (47.2 ± 1.2)

The relationship is statistically significant (p < 0.05).
```

#### 4. Coefficient of Variation Analysis
```
Coefficient of Variation Analysis
----------------------------------
Contact Duration CV:
  4mm: 1.30% (Excellent precision)
  40mm: 2.63% (Excellent precision)
  52mm: 3.06% (Excellent precision)

Peak Force CV:
  4mm: 1.11% (Excellent precision)
  40mm: 2.83% (Excellent precision)
  52mm: 4.07% (Excellent precision)

All measurements show CV < 5%, indicating excellent experimental precision.
```

#### 5. Summary
```
Summary
-------
- Duration increases linearly with thickness
- Peak force decreases linearly with thickness
- Both relationships are statistically significant
- Measurement precision is excellent (CV < 5%)
- Results support hypothesis that thicker pads increase contact time
  and reduce peak force
```

**Usage**:
- Quick reference for regression results
- Verification of statistical significance
- Assessment of data quality
- Inclusion in research reports

## Annotated Profiles

### Files: `annotated_profiles/<thickness>mm_<run>.png`

**Purpose**: Visual representation of force-time profiles with contact regions highlighted.

**Location**: `outputs/contact_analysis_<threshold>pct/annotated_profiles/`

**Format**: PNG images (300 DPI)

**Content**:
- Full force-time profile
- Shaded contact region (green/yellow)
- Vertical lines marking contact start and end
- Annotated values:
  - Contact duration (s)
  - Peak force (N)
  - Threshold force (N)
- Grid for easy reading
- Proper axis labels and units

**Example Filename**: `4mm_Run_1.png`

**Visualization Features**:
- **X-axis**: Time (s)
- **Y-axis**: Force (N)
- **Shaded region**: Contact duration period
- **Dotted lines**: Contact boundaries
- **Title**: Pad thickness and run identifier
- **Annotations**: Key metrics in text box

**Usage**:
- Visual verification of contact detection
- Quality control for individual runs
- Presentation in reports and papers
- Identifying anomalous runs

**Typical Appearance**:
```
Title: Force-Time Profile - 4mm Pad, Run 1

Force (N)
   50 |           *
      |         /   \
   40 |        /     \
      |       /       \
   30 |      |  XXXX  |        <- Shaded contact region
      |     /   XXXX   \
   20 |    |    XXXX    |
      |   /     XXXX     \
   10 |  |      XXXX      |
      | /       XXXX       \
    0 |/________XXXX________\___
       0    0.05  0.10  0.15  Time (s)
              ^         ^
              |         |
          Start(0.098s) End(0.1214s)

Annotations:
Duration: 0.0234 s
Peak Force: 45.2 N
Threshold: 2.26 N (5%)
```

## Aligned Data Files

### Files: `dr lee go brr - <thickness>mm_aligned.csv`

**Purpose**: Preprocessed force-time data with runs synchronized by peak alignment.

**Location**: `outputs/dr lee go brr - <thickness>mm/`

**Format**: CSV with multi-level column headers

**Structure**:
```csv
Run 1,Run 1,Run 2,Run 2,Run 3,Run 3
time_s,force_N,time_s,force_N,time_s,force_N
0.000,0.0,0.000,0.1,0.000,0.0
0.001,0.2,0.001,0.3,0.001,0.1
...
```

**Usage**:
- Input for contact duration analysis
- Comparison across runs
- Visual overlay plots
- Method validation

**Not Typically User-Generated**: Created by alignment preprocessing (external tool).

## File Size Estimates

| File Type | Typical Size | Notes |
|-----------|--------------|-------|
| `contact_duration_detailed.csv` | 5-20 KB | Depends on number of runs |
| `contact_duration_summary.csv` | 1-2 KB | One row per thickness |
| `summary_table_formatted.csv` | < 1 KB | Minimal formatting |
| `statistical_report.txt` | 2-5 KB | Text only |
| Annotated profile PNG | 100-300 KB each | 300 DPI, full resolution |
| Aligned data CSV | 500 KB - 5 MB | Depends on sampling rate and duration |

## Data Retention

### Recommended Workflow

1. **Keep all detailed CSV files**: Raw measurements
2. **Archive annotated profiles**: Visual records
3. **Version control statistical reports**: Track analysis changes
4. **Backup aligned data**: Input for reanalysis

### Long-term Storage

- **Essential**: Summary files, statistical reports
- **Important**: Detailed measurements, formatted tables
- **Archive**: Annotated profiles (can regenerate)
- **Optional**: Aligned data (large, can recreate if needed)

## Interpreting Results

### Contact Duration Trends

**Expected Pattern**: Duration increases with thickness

**Physical Interpretation**:
- Thicker pads compress more
- Longer deceleration phase
- Extended force application time

**Typical Slope**: 1-2 × 10⁻⁴ s/mm

### Peak Force Trends

**Expected Pattern**: Peak force decreases with thickness

**Physical Interpretation**:
- Impulse-momentum theorem: `J = F·Δt`
- Constant momentum change (same impact)
- Longer Δt → Lower F

**Typical Slope**: -0.3 to -0.5 N/mm

### Coefficient of Variation

**Excellent (CV < 5%)**: High-quality data, good experimental control

**Good (5-10%)**: Acceptable variability, minor improvements possible

**Moderate (10-20%)**: Consider improving experimental technique

**Poor (> 20%)**: Investigate systematic issues

### Statistical Significance

**p < 0.05**: Significant relationship, reject null hypothesis

**p ≥ 0.05**: No significant relationship, results inconclusive

**R² > 0.9**: Strong linear relationship, model fits data well

**R² < 0.7**: Weak relationship, consider non-linear models

## Troubleshooting Output Issues

### Missing Files

**Problem**: Expected output files not generated

**Possible Causes**:
- Input file path incorrect
- Insufficient data (< 2 runs)
- All runs failed validation
- Permissions issue

**Solution**:
- Check file paths in script
- Verify input data format
- Review console output for errors
- Ensure write permissions

### Unexpected Values

**Problem**: Results don't match expectations (e.g., negative durations)

**Possible Causes**:
- Incorrect threshold
- Misaligned data
- Invalid force readings
- Wrong units

**Solution**:
- Check threshold parameter
- Verify data alignment
- Inspect annotated profiles visually
- Confirm unit consistency

### Plot Issues

**Problem**: Annotated profiles show incorrect contact regions

**Possible Causes**:
- Threshold too high/low
- Noisy force signal
- Baseline drift
- Multiple impacts

**Solution**:
- Adjust threshold (try 3-10% range)
- Enable smoothing
- Inspect raw force data
- Use alternative detection method

### File Corruption

**Problem**: CSV files unreadable or malformed

**Possible Causes**:
- Script interrupted mid-write
- Disk full
- Encoding issues

**Solution**:
- Rerun analysis
- Check disk space
- Verify CSV encoding (UTF-8)
- Validate with spreadsheet software
