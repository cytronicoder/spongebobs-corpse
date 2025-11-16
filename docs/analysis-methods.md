# Analysis Methods

## Contact Duration Detection

### Overview

Contact duration is the time period during which an impacting object remains in physical contact with the viscoelastic pad. Accurate detection of this interval is critical for understanding impact attenuation behavior.

### Method 1: Force Threshold (Primary)

#### Principle
Contact is defined as the period when the magnitude of force exceeds a specified fraction of the peak force.

#### Algorithm

1. **Signal Smoothing** (optional):
   - Apply Savitzky-Golay filter to reduce noise
   - Window length: 11 points
   - Polynomial order: 3

2. **Peak Detection**:
   - Find maximum absolute force: `F_peak = max(|F(t)|)`

3. **Threshold Calculation**:
   - Threshold force: `F_threshold = α · F_peak`
   - Default: `α = 0.05` (5% of peak)

4. **Contact Start**:
   - Search backward from peak
   - Start when `|F(t)| > F_threshold`
   - Time point: `t_start`

5. **Contact End**:
   - Search forward from peak
   - End when `|F(t)| < F_threshold`
   - Time point: `t_end`

6. **Duration**:
   - `Δt = t_end - t_start`

#### Advantages
- Robust to noise (with smoothing)
- Intuitive physical interpretation
- Works with force data only
- Validated in literature

#### Limitations
- Threshold selection affects results
- May miss very brief contacts
- Sensitive to baseline drift

#### Implementation
```python
contact_info = calculate_contact_duration_threshold(
    time=time_array,
    force=force_array,
    threshold_fraction=0.05,  # 5% threshold
    smooth=True
)
```

### Method 2: Velocity-Based

#### Principle
Contact occurs between initial impact (velocity reversal) and rebound (second velocity reversal).

#### Algorithm

1. **Velocity Analysis**:
   - Smooth velocity signal
   - Find zero-crossings: `v(t) = 0`

2. **Impact Detection**:
   - First zero-crossing: deceleration to zero (compression begins)
   - Time: `t_impact`

3. **Rebound Detection**:
   - Second zero-crossing: acceleration from zero (compression ends)
   - Time: `t_rebound`

4. **Duration**:
   - `Δt = t_rebound - t_impact`

#### Advantages
- Physical interpretation (momentum conservation)
- Independent of force threshold
- Captures full compression-expansion cycle

#### Limitations
- Requires velocity data
- Sensitive to noise in velocity
- May miss partial contacts

### Method 3: Energy-Based

#### Principle
Contact period corresponds to kinetic energy transformation (conversion to elastic/heat energy and back).

#### Algorithm

1. **Energy Profile Analysis**:
   - Track kinetic energy: `E_k(t) = ½mv²(t)`

2. **Initial Energy**:
   - Maximum before impact: `E_0`

3. **Energy Minimum**:
   - Lowest kinetic energy (maximum compression)
   - Time: `t_min`

4. **Energy Recovery**:
   - Return to threshold fraction of `E_0`
   - Time: `t_recovery`

5. **Duration**:
   - `Δt = t_recovery - t_impact`

#### Advantages
- Accounts for energy dissipation
- Physical insight into energy transfer
- Complementary to force analysis

#### Limitations
- Requires energy calculation
- Assumes primarily kinetic energy
- Complex interpretation

## Statistical Methods

### Linear Regression

#### Standard Linear Regression

Uses `scipy.stats.linregress` for least-squares fitting:

**Model**: `y = mx + b`

**Outputs**:
- `m`: Slope
- `b`: Intercept
- `R²`: Coefficient of determination
- `p`: Statistical significance
- `SE`: Standard error

**Assumptions**:
- Linear relationship
- Independent observations
- Normally distributed errors
- Homoscedasticity (constant variance)

#### Weighted Regression

When measurement uncertainties (`σ_y`) are available:

**Weights**: `w_i = 1/σ_y_i²`

**Minimizes**: `Σ w_i(y_i - mx_i - b)²`

**Benefits**:
- Accounts for varying precision
- More reliable parameter estimates
- Better uncertainty propagation

### Uncertainty Estimation

#### Bootstrap Resampling

Used to estimate uncertainties in regression parameters:

**Algorithm**:
1. Perform `N_boot = 1000` iterations
2. For each iteration:
   - Randomly sample data with replacement
   - Add noise based on `yerr`: `y_boot = y + N(0, yerr)`
   - Calculate regression parameters: `m_boot`, `b_boot`
3. Calculate standard deviations:
   - `σ_m = std(m_boot)`
   - `σ_b = std(b_boot)`

**Advantages**:
- Non-parametric (no distribution assumptions)
- Robust to outliers
- Accounts for measurement uncertainties
- Provides empirical confidence intervals

#### Error Bar Method

Geometric approach using error bar extremes:

**Minimum Slope**:
- Connect `(x_1, y_1 + σ_1)` to `(x_n, y_n - σ_n)`
- `m_min = [(y_n - σ_n) - (y_1 + σ_1)] / (x_n - x_1)`

**Maximum Slope**:
- Connect `(x_1, y_1 - σ_1)` to `(x_n, y_n + σ_n)`
- `m_max = [(y_n + σ_n) - (y_1 - σ_1)] / (x_n - x_1)`

**Uncertainty**:
- `σ_m = (m_max - m_min) / 2`

**Use**: Visual representation of uncertainty range

### Confidence Intervals

#### 95% Confidence Band

For regression line predictions:

**Formula**:
```
CI(x) = t_{0.975, n-2} · SE · sqrt[1/n + (x - x̄)²/S_xx]
```

Where:
- `t_{0.975, n-2}`: t-distribution critical value (95% CI, n-2 degrees of freedom)
- `SE`: Standard error of regression
- `n`: Number of data points
- `x̄`: Mean of x values
- `S_xx = Σ(x_i - x̄)²`: Sum of squared deviations

**Interpretation**:
- 95% probability that true regression line lies within band
- Band widens at extremes (extrapolation uncertainty)
- Narrowest at mean x value

### Statistical Significance

#### Hypothesis Testing

**Null Hypothesis** (`H_0`): No relationship between variables (`m = 0`)

**Alternative Hypothesis** (`H_a`): Relationship exists (`m ≠ 0`)

**Test Statistic**:
```
t = m / SE_m
```

**P-value**: Probability of observing data if `H_0` is true

**Decision Rule**:
- `p < 0.05`: Reject `H_0` (significant relationship)
- `p ≥ 0.05`: Fail to reject `H_0` (no significant relationship)

#### Effect Size

**R² (Coefficient of Determination)**:
```
R² = 1 - SS_res / SS_tot
```

Where:
- `SS_res = Σ(y_i - ŷ_i)²`: Residual sum of squares
- `SS_tot = Σ(y_i - ȳ)²`: Total sum of squares

**Interpretation**:
- `R² = 0.9`: 90% of variance explained by model
- Measures goodness of fit
- Range: [0, 1], higher is better

### Coefficient of Variation

#### Definition

Relative measure of dispersion:
```
CV = (σ / μ) × 100%
```

Where:
- `σ`: Standard deviation
- `μ`: Mean

#### Interpretation

- **CV < 5%**: Excellent precision
- **5% ≤ CV < 10%**: Good precision
- **10% ≤ CV < 20%**: Moderate precision
- **CV ≥ 20%**: Poor precision

#### Use in Analysis

- Assesses measurement repeatability
- Compares precision across different thicknesses
- Identifies problematic experimental conditions
- Independent of units/magnitude

## Signal Processing

### Savitzky-Golay Filtering

#### Purpose
Smooth noisy force signals while preserving peak characteristics.

#### Method
Fits successive sub-sets of adjacent data points with low-degree polynomial via least squares.

#### Parameters
- **Window length** (`w = 11`): Number of points in window (must be odd)
- **Polynomial order** (`p = 3`): Degree of fitting polynomial

#### Advantages
- Preserves peak heights and widths
- Minimal phase distortion
- Suitable for derivatives
- Less smoothing than moving average

#### Implementation
```python
from scipy.signal import savgol_filter

smoothed_force = savgol_filter(force, window_length=11, polyorder=3)
```

### Data Validation

#### Quality Checks

1. **NaN/Inf Filtering**:
   - Remove non-finite values
   - Mask invalid data points
   - Interpolate small gaps if needed

2. **Outlier Detection**:
   - Identify points > 3σ from mean
   - Visual inspection of force profiles
   - Exclude clearly erroneous runs

3. **Alignment Verification**:
   - Check peak synchronization
   - Verify time array monotonicity
   - Ensure consistent sampling rate

#### Minimum Requirements

- **Points per run**: ≥ 100 for reliable contact detection
- **Runs per thickness**: ≥ 3 for statistical analysis
- **Sampling rate**: ≥ 1000 Hz (recommended)
- **Signal-to-noise ratio**: Peak force > 10× baseline noise

## Physical Models

### Viscoelastic Theory

#### Spring-Dashpot Model

Viscoelastic behavior modeled as:
- **Spring** (elastic): `F_elastic = k·x`
- **Dashpot** (viscous): `F_viscous = c·v`

**Combined**: `F = k·x + c·v`

#### Impact Response

**Thicker pads**:
- Greater compression distance
- Longer contact duration
- Lower peak force (impulse distributed over time)

**Impulse-Momentum Theorem**:
```
J = ∫F dt = Δp
```

For constant momentum change:
- Longer `Δt` → Lower average `F`

### Energy Dissipation

#### Energy Balance

**Initial kinetic energy**: `E_0 = ½mv_0²`

**Energy partitioning**:
- Elastic storage (recoverable)
- Viscous dissipation (heat)
- Plastic deformation (permanent)

**Coefficient of Restitution**:
```
e = v_rebound / v_impact
```

**Energy loss**:
```
E_loss = E_0(1 - e²)
```

#### Attenuation Mechanism

Thicker pads increase energy dissipation through:
1. **Greater compression**: More material deformation
2. **Longer contact time**: Extended viscous dissipation
3. **Increased damping**: Material hysteresis effects

## Experimental Considerations

### Error Sources

1. **Systematic Errors**:
   - Sensor calibration drift
   - Alignment inconsistencies
   - Environmental temperature effects

2. **Random Errors**:
   - Electrical noise
   - Impact velocity variations
   - Material property variations

### Uncertainty Propagation

#### Contact Duration Uncertainty

Dominant sources:
- Force threshold selection: ±10-20% variation
- Sampling resolution: ±1 time step
- Signal noise: Mitigated by smoothing

#### Regression Uncertainty

Propagated through:
- Bootstrap resampling (parameter uncertainty)
- Confidence intervals (prediction uncertainty)
- Error bars (measurement uncertainty)

### Validation Strategies

1. **Method Comparison**: Threshold vs. velocity vs. energy
2. **Threshold Sensitivity**: Test range 3-10%
3. **Repeatability**: Multiple runs per condition
4. **Physical Plausibility**: Check against expected trends
