# Examples

## Example 1: Basic Analysis Pipeline

### Objective
Process aligned data from three pad thicknesses and generate summary statistics.

### Steps

#### 1. Verify Data Files

```bash
ls -lh data/
```

Expected output:
```
dr lee go brr - 4mm.csv
dr lee go brr - 40mm.csv
dr lee go brr - 52mm.csv
```

#### 2. Process Contact Duration (5% Threshold)

```bash
# Process 4mm data
python contact_duration.py "outputs/dr lee go brr - 4mm/dr lee go brr - 4mm_aligned.csv"

# Process 40mm data
python contact_duration.py "outputs/dr lee go brr - 40mm/dr lee go brr - 40mm_aligned.csv"

# Process 52mm data
python contact_duration.py "outputs/dr lee go brr - 52mm/dr lee go brr - 52mm_aligned.csv"
```

#### 3. Run Statistical Analysis

```bash
python analysis.py
```

#### 4. Check Results

```bash
ls -lh outputs/contact_analysis_5pct/
```

Expected files:
- `contact_duration_detailed.csv`
- `contact_duration_summary.csv`
- `summary_table_formatted.csv`
- `statistical_report.txt`
- `annotated_profiles/` directory

### Expected Results

**Duration vs. Thickness**:
- Positive correlation (R² > 0.95)
- Slope ≈ 1-2 × 10⁻⁴ s/mm
- Statistical significance (p < 0.05)

**Peak Force vs. Thickness**:
- Negative correlation (R² > 0.95)
- Slope ≈ -0.3 to -0.5 N/mm
- Statistical significance (p < 0.05)

## Example 2: Threshold Sensitivity Analysis

### Objective
Compare results using different threshold values (3%, 5%, 10%).

### Steps

#### Process with 3% Threshold

```bash
python contact_duration.py "outputs/dr lee go brr - 4mm/dr lee go brr - 4mm_aligned.csv" --threshold 0.03
python contact_duration.py "outputs/dr lee go brr - 40mm/dr lee go brr - 40mm_aligned.csv" --threshold 0.03
python contact_duration.py "outputs/dr lee go brr - 52mm/dr lee go brr - 52mm_aligned.csv" --threshold 0.03
```

Results in: `outputs/contact_analysis_3pct/`

#### Process with 10% Threshold

```bash
python contact_duration.py "outputs/dr lee go brr - 4mm/dr lee go brr - 4mm_aligned.csv" --threshold 0.10
python contact_duration.py "outputs/dr lee go brr - 40mm/dr lee go brr - 40mm_aligned.csv" --threshold 0.10
python contact_duration.py "outputs/dr lee go brr - 52mm/dr lee go brr - 52mm_aligned.csv" --threshold 0.10
```

Results in: `outputs/contact_analysis_10pct/`

#### Compare Results

```python
import pandas as pd

# Load summaries
summary_3pct = pd.read_csv("outputs/contact_analysis_3pct/contact_duration_summary.csv")
summary_5pct = pd.read_csv("outputs/contact_analysis_5pct/contact_duration_summary.csv")
summary_10pct = pd.read_csv("outputs/contact_analysis_10pct/contact_duration_summary.csv")

# Compare durations
print("Duration comparison (4mm pad):")
print(f"3% threshold: {summary_3pct[summary_3pct.thickness_mm == 4.0]['duration_s_mean'].values[0]:.4f} s")
print(f"5% threshold: {summary_5pct[summary_5pct.thickness_mm == 4.0]['duration_s_mean'].values[0]:.4f} s")
print(f"10% threshold: {summary_10pct[summary_10pct.thickness_mm == 4.0]['duration_s_mean'].values[0]:.4f} s")
```

### Expected Observations

- Lower threshold → Longer duration (captures more of impact tail)
- Higher threshold → Shorter duration (focuses on main impact)
- Trend (slope) should remain similar across thresholds
- CV values should be comparable

## Example 3: Method Comparison

### Objective
Validate threshold method against velocity and energy methods.

### Prerequisites
Aligned data must include velocity and energy columns.

### Steps

#### Threshold Method (Default)

```bash
python contact_duration.py "outputs/dr lee go brr - 4mm/dr lee go brr - 4mm_aligned.csv" --method threshold
```

#### Velocity Method

```bash
python contact_duration.py "outputs/dr lee go brr - 4mm/dr lee go brr - 4mm_aligned.csv" --method velocity
```

#### Energy Method

```bash
python contact_duration.py "outputs/dr lee go brr - 4mm/dr lee go brr - 4mm_aligned.csv" --method energy
```

#### Compare Results

```python
import pandas as pd

# Load detailed results
threshold_data = pd.read_csv("outputs/contact_analysis_5pct/contact_duration_detailed.csv")
threshold_data = threshold_data[threshold_data['method'] == 'threshold']

# If other methods saved to same file or separate files
# Load and compare durations for same runs
```

### Expected Observations

- Methods should give similar durations (within 10-20%)
- Threshold method typically most robust
- Velocity method sensitive to noise
- Energy method may differ if significant energy dissipation

## Example 4: Custom Visualization

### Objective
Create custom plots from analysis results.

### Script: `custom_plot.py`

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
summary = pd.read_csv("outputs/contact_analysis_5pct/contact_duration_summary.csv")
detailed = pd.read_csv("outputs/contact_analysis_5pct/contact_duration_detailed.csv")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Duration distribution by thickness
for thickness in summary['thickness_mm'].unique():
    data = detailed[detailed['thickness_mm'] == thickness]['duration_s']
    ax1.hist(data, alpha=0.6, label=f'{thickness}mm', bins=10)

ax1.set_xlabel('Contact Duration (s)')
ax1.set_ylabel('Frequency')
ax1.set_title('Contact Duration Distribution by Thickness')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Peak force distribution
for thickness in summary['thickness_mm'].unique():
    data = detailed[detailed['thickness_mm'] == thickness]['peak_force_N']
    ax2.hist(data, alpha=0.6, label=f'{thickness}mm', bins=10)

ax2.set_xlabel('Peak Force (N)')
ax2.set_ylabel('Frequency')
ax2.set_title('Peak Force Distribution by Thickness')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('custom_distributions.png', dpi=300)
plt.show()
```

### Run

```bash
python custom_plot.py
```

## Example 5: Extracting Specific Run Data

### Objective
Extract and analyze a specific run from detailed results.

### Python Script

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load detailed data
detailed = pd.read_csv("outputs/contact_analysis_5pct/contact_duration_detailed.csv")

# Filter for specific thickness and run
run_data = detailed[(detailed['thickness_mm'] == 4.0) & (detailed['run'] == 'Run 1')]

print("Run 1 - 4mm Pad:")
print(f"Duration: {run_data['duration_s'].values[0]:.4f} s")
print(f"Peak Force: {run_data['peak_force_N'].values[0]:.2f} N")
print(f"Start Time: {run_data['start_time_s'].values[0]:.4f} s")
print(f"End Time: {run_data['end_time_s'].values[0]:.4f} s")
print(f"Threshold Force: {run_data['threshold_force_N'].values[0]:.2f} N")
```

## Example 6: Batch Processing Script

### Objective
Automate processing for all thicknesses.

### Script: `batch_process.sh`

```bash
#!/bin/bash

# Define thicknesses
thicknesses=("4mm" "40mm" "52mm")

# Define threshold
threshold=0.05

# Process each thickness
for thickness in "${thicknesses[@]}"; do
    echo "Processing $thickness data..."
    python contact_duration.py \
        "outputs/dr lee go brr - $thickness/dr lee go brr - ${thickness}_aligned.csv" \
        --threshold $threshold
done

# Run analysis
echo "Running statistical analysis..."
python analysis.py

echo "Processing complete! Results in outputs/contact_analysis_$(echo "$threshold*100" | bc | cut -d'.' -f1)pct/"
```

### Run

```bash
chmod +x batch_process.sh
./batch_process.sh
```

## Example 7: Statistical Report Interpretation

### Sample Report Section

```
Linear Regression: Duration vs Thickness
-----------------------------------------
Slope: (1.23 ± 0.04) × 10⁻⁴ s/mm
Intercept: (0.0198 ± 0.0015) s
R² = 0.987
p-value = 0.0023
```

### Interpretation

**Slope**: For every 1mm increase in pad thickness:
- Contact duration increases by 1.23 × 10⁻⁴ s (0.000123 s)
- Uncertainty: ± 0.04 × 10⁻⁴ s

**For 48mm difference (52mm - 4mm)**:
- Expected duration increase: 48 × 1.23 × 10⁻⁴ = 0.0059 s
- Observed: ≈ 0.0358 s (from 0.0231 to 0.0589)

**R² = 0.987**: Model explains 98.7% of variance

**p = 0.0023**: Highly significant (p < 0.01), strong evidence against null hypothesis

### Conclusion Template

```
The contact duration increases linearly with pad thickness 
(R² = 0.987, p = 0.0023), with a slope of (1.23 ± 0.04) × 10⁻⁴ s/mm. 
This confirms the hypothesis that thicker viscoelastic pads increase 
contact time, thereby reducing peak impact forces through the 
impulse-momentum theorem.
```

## Example 8: Identifying Outliers

### Objective
Detect and investigate anomalous runs.

### Python Script

```python
import pandas as pd
import numpy as np

# Load detailed data
detailed = pd.read_csv("outputs/contact_analysis_5pct/contact_duration_detailed.csv")

# Group by thickness
for thickness in detailed['thickness_mm'].unique():
    data = detailed[detailed['thickness_mm'] == thickness]
    
    # Calculate statistics
    mean_duration = data['duration_s'].mean()
    std_duration = data['duration_s'].std()
    
    # Identify outliers (> 2 standard deviations)
    outliers = data[np.abs(data['duration_s'] - mean_duration) > 2 * std_duration]
    
    if not outliers.empty:
        print(f"\nOutliers detected for {thickness}mm:")
        print(outliers[['run', 'duration_s', 'peak_force_N']])
    else:
        print(f"\nNo outliers for {thickness}mm")
```

### Investigation Steps

1. **Check annotated profile**: Visual inspection of force-time curve
2. **Compare with other runs**: Look for systematic differences
3. **Review experimental notes**: Equipment issues, environmental factors
4. **Decision**: Keep (valid variation) or exclude (error)

## Example 9: Exporting for Publication

### Objective
Prepare results for inclusion in research paper.

### LaTeX Table

```latex
\begin{table}[h]
\centering
\caption{Contact Duration and Peak Force vs. Pad Thickness}
\label{tab:results}
\begin{tabular}{ccccc}
\hline
\textbf{Thickness} & \textbf{Duration} & \textbf{CV} & \textbf{Peak Force} & \textbf{CV} \\
\textbf{(mm)} & \textbf{(s)} & \textbf{(\%)} & \textbf{(N)} & \textbf{(\%)} \\
\hline
4 & $0.0231 \pm 0.0003$ & 1.30 & $45.0 \pm 0.5$ & 1.11 \\
40 & $0.0456 \pm 0.0012$ & 2.63 & $28.3 \pm 0.8$ & 2.83 \\
52 & $0.0589 \pm 0.0018$ & 3.06 & $22.1 \pm 0.9$ & 4.07 \\
\hline
\end{tabular}
\end{table}
```

### Figures

```bash
# Copy plots to paper directory
cp outputs/contact_analysis_5pct/duration_vs_thickness.png paper/figures/
cp outputs/contact_analysis_5pct/force_vs_thickness.png paper/figures/
```

### Figure Caption Template

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{figures/duration_vs_thickness.png}
\caption{Contact duration vs. pad thickness. Individual runs shown as 
gray points, means with error bars (±1 SD) in blue. Orange line shows 
linear regression with 95\% confidence interval (shaded). Pink dotted 
lines indicate uncertainty bounds based on error bars. 
Linear relationship: $t_c = (1.23 \pm 0.04) \times 10^{-4}d + (0.0198 \pm 0.0015)$, 
where $t_c$ is contact duration (s) and $d$ is thickness (mm). 
$R^2 = 0.987$, $p = 0.0023$.}
\label{fig:duration}
\end{figure}
```

## Example 10: Reanalysis with Different Parameters

### Scenario
Initial analysis used 5% threshold, but reviewer suggests 3% for better sensitivity.

### Quick Reanalysis

```bash
# Reprocess all data with 3% threshold
for file in outputs/dr\ lee\ go\ brr\ -\ */dr\ lee\ go\ brr\ -\ *_aligned.csv; do
    python contact_duration.py "$file" --threshold 0.03
done

# Run analysis
python analysis.py

# Compare results
echo "5% threshold results:"
cat outputs/contact_analysis_5pct/statistical_report.txt | grep "Slope:"

echo -e "\n3% threshold results:"
cat outputs/contact_analysis_3pct/statistical_report.txt | grep "Slope:"
```

### Response to Reviewer

```
We re-analyzed the data using a 3% threshold as suggested. The slope 
changed from (1.23 ± 0.04) × 10⁻⁴ s/mm to (1.31 ± 0.05) × 10⁻⁴ s/mm, 
consistent within uncertainty ranges. The R² value improved slightly 
from 0.987 to 0.991, and the relationship remains highly significant 
(p < 0.01). This confirms the robustness of our findings across 
different threshold values.
```

## Common Workflows Summary

### Standard Analysis
1. Process all thicknesses with contact_duration.py
2. Run analysis.py for statistics
3. Review statistical_report.txt
4. Check annotated_profiles for quality

### Quality Control
1. Calculate CV values from summary
2. Identify outliers in detailed data
3. Inspect annotated profiles visually
4. Exclude/investigate anomalous runs

### Publication Preparation
1. Export formatted summary table
2. Copy regression plots
3. Extract key statistics from report
4. Prepare figure captions with equations

### Method Validation
1. Compare different detection methods
2. Test threshold sensitivity
3. Verify alignment quality
4. Cross-check with theoretical predictions
