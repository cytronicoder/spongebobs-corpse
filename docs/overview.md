# Project Overview

## Investigating Impact Attenuation Through Variable-Thickness Viscoelastic Pads

This project is part of an IBDP Physics HL Internal Assessment investigating how viscoelastic materials of varying thickness attenuate impact forces.

## Research Question

How does the thickness of viscoelastic pads affect their ability to attenuate impact forces, as measured by contact duration and peak force reduction?

## Experimental Design

### Materials
- Viscoelastic foam pads at different thicknesses: 4mm, 40mm, and 52mm
- Force sensor apparatus for measuring impact forces
- Data acquisition system recording force vs. time at high frequency

### Measurements
- **Force-time profiles**: Continuous recording of force during impact events
- **Contact duration**: Time period during which the impacting object remains in contact with the pad
- **Peak force**: Maximum force experienced during impact
- **Multiple trials**: Repeated measurements for statistical validity

### Variables
- **Independent variable**: Pad thickness (mm)
- **Dependent variables**: 
  - Contact duration (s)
  - Peak force (N)
  - Coefficient of variation (CV)
- **Controlled variables**: Impact velocity, mass of impacting object, environmental conditions

## Analysis Approach

### Data Processing
1. **Alignment**: Synchronize multiple experimental runs by aligning peak force events
2. **Noise reduction**: Apply Savitzky-Golay filtering to smooth force signals
3. **Contact detection**: Implement multiple methods (threshold-based, velocity-based, energy-based)
4. **Statistical analysis**: Calculate means, standard deviations, and perform linear regression

### Visualization
- Individual run scatter plots with mean values
- Linear regression with 95% confidence intervals
- Min/max slope uncertainty bounds
- Annotated force-time profiles showing contact regions

### Statistical Methods
- **Linear regression**: Examine relationship between thickness and dependent variables
- **Bootstrap resampling**: Estimate uncertainties in regression parameters
- **T-tests**: Compare means between different thickness groups
- **Coefficient of variation**: Assess measurement precision

## Key Features

### Multiple Contact Detection Methods
The analysis implements three independent methods for determining contact duration:

1. **Force Threshold Method** (primary): Contact defined as period when force exceeds a threshold (typically 5% of peak force)
2. **Velocity-based Method**: Contact determined by velocity reversal points
3. **Energy-based Method**: Contact identified through kinetic energy changes

### Robust Statistical Analysis
- Bootstrap resampling (1000 iterations) for parameter uncertainty estimation
- Weighted regression accounting for measurement uncertainties
- Confidence interval calculation using Student's t-distribution
- Comprehensive error propagation

### Flexible Data Processing
- Automatic alignment of multiple experimental runs
- Configurable smoothing and filtering parameters
- Support for various threshold values
- Batch processing capabilities

## Output Products

### Quantitative Results
- Summary statistics tables (CSV format)
- Detailed run-by-run measurements (CSV format)
- Statistical test reports (text format)

### Visualizations
- Scatter plots with regression analysis
- Annotated force-time profiles
- Comparison plots across thicknesses
- Uncertainty visualization

### Quality Metrics
- Coefficient of variation for each measurement set
- R² values for regression fits
- P-values for statistical significance
- Uncertainty estimates for all fitted parameters

## Physical Context

### Viscoelastic Behavior
Viscoelastic materials exhibit both viscous and elastic characteristics:
- **Elastic response**: Energy storage and recovery
- **Viscous response**: Energy dissipation through internal friction

### Impact Attenuation Mechanism
Thicker pads generally:
- Increase contact duration (force applied over longer time)
- Reduce peak force (impulse distributed over extended period)
- Increase energy absorption capacity

### Engineering Applications
Understanding impact attenuation is critical for:
- Protective equipment design (helmets, padding)
- Packaging materials
- Vehicle crash safety systems
- Sports equipment
- Shock absorption systems

## Software Design Principles

### Modularity
- Separate utilities module for shared functions
- Independent contact detection methods
- Reusable plotting and statistical functions

### Reproducibility
- Fixed random seeds for consistent results
- Complete parameter documentation
- Version-controlled analysis pipeline

### Extensibility
- Easy addition of new contact detection methods
- Flexible configuration options
- Support for different data formats and experimental designs
