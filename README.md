### Investigating Impact Attenuation Through Variable-Thickness Viscoelastic Pads

> [!NOTE]
> This repository contains the source code and data for my IBDP Physics HL Internal Assessment. It is not intended for public use, and should be used solely for educational purposes. However, you are free to explore the code and data for learning on your own.

We investigated how viscoelastic pads of varying thickness attenuate impacts. We processed force-time data from impact tests on viscoelastic materials, calculated contact durations, and performed statistical evaluations to determine the effectiveness of different pad thicknesses in reducing impact forces.

[![Results](/outputs/contact_analysis_10pct/comprehensive_summary.png)](/outputs/contact_analysis_10pct/comprehensive_summary.png)

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
2. Run the analysis:

   ```bash
   python analysis.py
   ```

3. View results in the `outputs/` directory, including plots and statistical reports.

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
