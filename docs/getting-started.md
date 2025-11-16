# Getting Started

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Clone the Repository

```bash
git clone https://github.com/cytronicoder/spongebobs-corpse.git
cd spongebobs-corpse
```

### Install Dependencies

Install all required Python packages:

```bash
pip install numpy pandas matplotlib scipy
```

Or use a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib scipy
```

### Verify Installation

Test that all imports work correctly:

```bash
python -c "import numpy, pandas, matplotlib, scipy; print('All dependencies installed successfully')"
```

## Running the Analysis

### Basic Usage

#### Step 1: Prepare Data
Place your CSV data files in the `data/` directory. Files should follow the naming convention that includes thickness information (e.g., `experiment - 4mm.csv`).

#### Step 2: Run Analysis Script

```bash
python analysis.py
```

This will process all aligned data files found in the `outputs/` directory and generate statistical summaries and plots.

#### Step 3: View Results
Check the `outputs/contact_analysis_<threshold>/` directory for:
- Summary statistics tables
- Detailed measurements
- Statistical reports
- Visualization plots

### Advanced Usage

#### Contact Duration Analysis

Process raw force-time data to calculate contact durations:

```bash
python contact_duration.py <aligned_data_file.csv> --threshold 0.05 --method threshold
```

**Parameters:**
- `aligned_data_file.csv`: Path to aligned force-time data file
- `--threshold`: Force threshold fraction (default: 0.05 = 5% of peak force)
- `--method`: Contact detection method: `threshold`, `velocity`, or `energy`
- `--smooth`: Enable signal smoothing (default: enabled)

**Example:**

```bash
python contact_duration.py outputs/dr\ lee\ go\ brr\ -\ 4mm/dr\ lee\ go\ brr\ -\ 4mm_aligned.csv --threshold 0.03 --method threshold
```

#### Custom Analysis Parameters

Modify parameters in the scripts for custom analysis:

**In `analysis.py`:**
```python
# Adjust bootstrap iterations for uncertainty estimation
n_boot = 1000  # Increase for more precise uncertainties

# Modify plot aesthetics
colors = {
    "gray": "#7f8c8d",
    "blue": "#3498db",
    "orange": "#e67e22",
    "pink": "#e74c3c"
}
```

**In `contact_duration.py`:**
```python
# Change smoothing parameters
window_length = 11  # Must be odd
polyorder = 3       # Polynomial order for Savitzky-Golay filter

# Adjust threshold
threshold_fraction = 0.05  # 5% of peak force
```

## Workflow Examples

### Example 1: Standard Analysis Pipeline

```bash
# 1. Process aligned data with 5% threshold
python contact_duration.py outputs/dr\ lee\ go\ brr\ -\ 4mm/dr\ lee\ go\ brr\ -\ 4mm_aligned.csv

# 2. Repeat for other thicknesses
python contact_duration.py outputs/dr\ lee\ go\ brr\ -\ 40mm/dr\ lee\ go\ brr\ -\ 40mm_aligned.csv
python contact_duration.py outputs/dr\ lee\ go\ brr\ -\ 52mm/dr\ lee\ go\ brr\ -\ 52mm_aligned.csv

# 3. Run comprehensive statistical analysis
python analysis.py
```

### Example 2: Compare Different Thresholds

```bash
# Process with 3% threshold
python contact_duration.py outputs/dr\ lee\ go\ brr\ -\ 4mm/dr\ lee\ go\ brr\ -\ 4mm_aligned.csv --threshold 0.03

# Process with 10% threshold
python contact_duration.py outputs/dr\ lee\ go\ brr\ -\ 4mm/dr\ lee\ go\ brr\ -\ 4mm_aligned.csv --threshold 0.10

# Compare results in outputs directories
```

### Example 3: Different Detection Methods

```bash
# Force threshold method (primary)
python contact_duration.py outputs/dr\ lee\ go\ brr\ -\ 4mm/dr\ lee\ go\ brr\ -\ 4mm_aligned.csv --method threshold

# Velocity-based method
python contact_duration.py outputs/dr\ lee\ go\ brr\ -\ 4mm/dr\ lee\ go\ brr\ -\ 4mm_aligned.csv --method velocity

# Energy-based method
python contact_duration.py outputs/dr\ lee\ go\ brr\ -\ 4mm/dr\ lee\ go\ brr\ -\ 4mm_aligned.csv --method energy
```

## Directory Structure After Running

```
spongebobs-corpse/
├── data/
│   ├── dr lee go brr - 4mm.csv      # Raw data
│   ├── dr lee go brr - 40mm.csv
│   └── dr lee go brr - 52mm.csv
├── outputs/
│   ├── contact_analysis_5pct/        # Results with 5% threshold
│   │   ├── contact_duration_detailed.csv
│   │   ├── contact_duration_summary.csv
│   │   ├── statistical_report.txt
│   │   ├── summary_table_formatted.csv
│   │   └── annotated_profiles/       # Force-time plots
│   ├── dr lee go brr - 4mm/          # Aligned data
│   │   ├── dr lee go brr - 4mm_aligned.csv
│   │   └── ...
│   └── ...
├── analysis.py
├── contact_duration.py
├── utils.py
└── README.md
```

## Next Steps

1. Review [Data Format](data-format.md) to understand input requirements
2. Check [Scripts Documentation](scripts.md) for detailed script usage
3. Learn about [Analysis Methods](analysis-methods.md) for methodology details
4. See [Output Files](output-files.md) for interpreting results
5. Consult [Examples](examples.md) for common use cases
