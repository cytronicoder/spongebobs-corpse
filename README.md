### Investigating Impact Attenuation Through Variable-Thickness Viscoelastic Pads

We investigated how viscoelastic pads of varying thickness attenuate impacts. We processed force-time data from impact tests on viscoelastic materials, calculated contact durations, and performed statistical evaluations to determine the effectiveness of different pad thicknesses in reducing impact forces.

[![Results](/outputs/contact_analysis_10pct/comprehensive_summary.png)](/outputs/contact_analysis_10pct/comprehensive_summary.png)

> [!NOTE]
> This repository contains the source code and data for my IBDP Physics HL Internal Assessment. It is not intended for public use, and should be used solely for educational purposes. However, you are free to explore the code and data for learning on your own.

#### Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[Getting Started](docs/getting-started.md)** - Installation and usage instructions
- **[Data Format](docs/data-format.md)** - Input data structure and requirements
- **[Scripts](docs/scripts.md)** - Detailed script documentation
- **[API Reference](docs/api-reference.md)** - Function and class documentation
- **[Analysis Methods](docs/analysis-methods.md)** - Mathematical and statistical methods
- **[Output Files](docs/output-files.md)** - Description of generated outputs
- **[Examples](docs/examples.md)** - Usage examples and workflows
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

#### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/cytronicoder/spongebobs-corpse.git
   cd spongebobs-corpse
   ```

2. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib scipy
   ```

#### Usage

1. Place your CSV data files in the `data/` directory.

2. Run the offset alignment:

   ```bash
   python offset.py
   ```

3. Calculate contact durations using multiple methods:

   ```bash
   # Use force-threshold method (default)
   python contact_duration.py --method threshold --threshold 0.05

   # Compare all three methods
   python contact_duration.py --method all

   # Use velocity-based method with custom threshold
   python contact_duration.py --method velocity --velocity_threshold 0.1

   # Use energy-based method with custom mass
   python contact_duration.py --method energy --cart_mass 0.5
   ```

4. Generate method comparison plots:

   ```bash
   python compare_methods.py "outputs/dr lee go brr - 40mm/dr lee go brr - 40mm_aligned.csv" \
       --run "Run 2" --threshold 0.05 --mass 0.5 --output "method_comparison.png"
   ```

5. Compare threshold sensitivity:

   ```bash
   python compare_thresholds.py "outputs/dr lee go brr - 40mm/dr lee go brr - 40mm_aligned.csv" \
       --run "Run 2" --thresholds 0.01 0.03 0.05 0.10 --output "threshold_comparison.png"
   ```

6. Perform statistical analysis:

   ```bash
   python analysis.py
   ```

7. View results in the `outputs/` directory, including plots and statistical reports.

#### Analysis Methods

The project implements three contact duration detection methods:

- **Force-Threshold (Primary):** Detects contact when |force| exceeds a percentage of peak force
- **Velocity-Based:** Uses velocity threshold detection to identify contact boundaries (handles sparse data)
- **Energy-Based:** Tracks kinetic energy transformation during impact (finds global maximum first)

See [Analysis Methods](docs/analysis-methods.md) for detailed mathematical descriptions.

#### Data Format

Input data should be CSV files with columns for time and force measurements. Example files are provided in the `data/` directory.

#### Requirements

- Python 3.8 or higher
- numpy
- pandas
- matplotlib
- scipy

#### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
