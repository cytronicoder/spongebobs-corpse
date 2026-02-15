# Statistical Methods Documentation

## Overview

This analysis has been upgraded from simple error-bar overlap comparisons to rigorous statistical inference with:

1. **Regression with Parameter Uncertainties**: Full confidence intervals on slope and intercept
2. **Hypothesis Tests**: Welch's t-tests for pairwise comparisons between adjacent thickness levels
3. **Clear Error Bar Definitions**: Explicit documentation of what uncertainties represent
4. **Multiple Comparison Correction**: Bonferroni correction for family-wise error control

## Error Bar Definitions

Throughout this analysis, error bars and uncertainties represent different types of variability:

### Standard Deviation (SD)
- **What it is**: Measure of spread in individual measurements
- **Formula**: `SD = sqrt(Σ(xi - mean)² / (n-1))`
- **Interpretation**: Describes the natural variability in your measurements
- **When to use**: To show the range of individual observations

### Standard Error (SE)
- **What it is**: Uncertainty in the estimated mean
- **Formula**: `SE = SD / sqrt(n)`
- **Interpretation**: How precisely we know the true mean value
- **When to use**: To show uncertainty in summary statistics (means)

### Combined Uncertainty
- **What it is**: Total uncertainty accounting for both random and systematic errors
- **Formula**: `u_combined = sqrt(u_random² + u_instrument²)`
  - `u_random = SE` (statistical uncertainty)
  - `u_instrument` = known instrument uncertainty (time: 0.01 s, force: 0.05 N)
- **Interpretation**: Best estimate of total measurement uncertainty
- **When to use**: In plots showing experimental results with error bars

### 95% Confidence Interval (CI)
- **What it is**: Range likely to contain the true population parameter
- **Formula**: `CI = estimate ± t(α/2, df) × SE`
  - `t(α/2, df)` is the critical t-value for 95% confidence
- **Interpretation**: 95% of CIs computed this way will contain the true value
- **When to use**: For regression parameters and pairwise comparisons

## Regression Analysis

### Linear Regression with Parameter Uncertainties

Instead of just reporting R² and p-values, we now compute:

#### Slope Uncertainty
- **Standard Error**: `SE(slope) = RMSE / sqrt(Σ(xi - mean_x)²)`
- **95% CI**: `slope ± t(α/2, n-2) × SE(slope)`
- **Interpretation**: If the CI excludes zero, the slope is significantly different from zero

#### Intercept Uncertainty
- **Standard Error**: `SE(intercept) = RMSE × sqrt(1/n + mean_x²/Σ(xi - mean_x)²)`
- **95% CI**: `intercept ± t(α/2, n-2) × SE(intercept)`

#### Example Interpretation
```
Slope (s_F): -0.035343 ± 0.002905
95% CI: [-0.042812, -0.027875]
p-value: p < 0.001***
```

**What this means**:
- Peak force decreases by 0.0353 N per mm of thickness
- We are 95% confident the true slope is between -0.0428 and -0.0279
- The relationship is highly significant (p < 0.001)
- The CI does not include zero, confirming a real negative trend

### Confidence Bands on Regression Line

The shaded bands on regression plots represent 95% confidence intervals for the **mean** response at each x value:

- **Confidence Interval**: Where we expect the true regression line to be
- **Formula**: `CI(mean) = ŷ ± t × RMSE × sqrt(1/n + (x - mean_x)²/Σ(xi - mean_x)²)`
- **Note**: Prediction intervals (for individual observations) are wider and not shown

### Model Selection

We compare models using:
- **R²**: Proportion of variance explained (0-1, higher is better)
- **Adjusted R²**: R² penalized for model complexity
- **RMSE**: Root Mean Squared Error (lower is better)

**Example**:
```
Peak Force: Linear model preferred (R²=0.9891 vs 0.9549)
Contact Duration: Power-law model preferred (R²=0.7653 vs 0.6269)
```

## Pairwise Comparisons (Hypothesis Tests)

### Welch's t-test

For each pair of adjacent thickness levels, we perform Welch's t-test, which:

- **Does not assume equal variances** (more robust than Student's t-test)
- Tests the null hypothesis: H₀: μ₁ = μ₂ (means are equal)
- Computes p-value: probability of observing this difference if H₀ is true

#### Test Statistic
```
t = (mean₁ - mean₂) / sqrt(SE₁² + SE₂²)
```

#### Degrees of Freedom (Welch-Satterthwaite)
```
df = (SE₁² + SE₂²)² / (SE₁⁴/(n₁-1) + SE₂⁴/(n₂-1))
```

#### 95% Confidence Interval for Difference
```
CI = (mean₁ - mean₂) ± t(α/2, df) × sqrt(SE₁² + SE₂²)
```

### Interpretation Guidelines

| p-value | Symbol | Interpretation |
|---------|--------|----------------|
| < 0.001 | ***    | Extremely strong evidence against H₀ |
| < 0.01  | **     | Strong evidence against H₀ |
| < 0.05  | *      | Moderate evidence against H₀ |
| ≥ 0.05  | ns     | Not significant, insufficient evidence |

### Example Interpretation

```
16.0 mm vs 28.0 mm:
  Mean difference: -0.012000 ± 0.004690
  95% CI: [-0.023218, -0.000782]
  t(6.6) = -2.558, p = 0.0394 *
```

**What this means**:
- Contact duration is 0.012 s shorter at 16 mm than at 28 mm
- We are 95% confident the true difference is between -0.023 and -0.001 s
- The CI does not include zero, indicating a real difference
- The difference is statistically significant (p = 0.039 < 0.05)

### Multiple Comparison Correction

When performing multiple tests (6 comparisons for 7 thickness levels), the chance of false positives increases. We apply **Bonferroni correction**:

- **Adjusted α**: `α_corrected = α / n_tests = 0.05 / 6 = 0.0083`
- **More conservative**: Only declare significance if p < 0.0083
- **Trade-off**: Reduces false positives but may miss real effects

**Example**:
```
Bonferroni-corrected alpha for Peak Force: 0.0083
Significant comparisons after correction: 2/6
```

This means only 2 out of 6 pairwise comparisons remain significant after accounting for multiple testing.

## Coefficient of Variation (Repeatability)

### Formula
```
CV = (SD / mean) × 100%
```

### Interpretation
- **CV < 5%**: Excellent repeatability (all our measurements!)
- **CV 5-10%**: Good repeatability
- **CV > 10%**: High variability, potential issues with method

### Example
```
Mean CV: 2.89%
Range: 2.14% - 4.72%
Measurements with CV ≤ 5%: 7/7 (100.0%)
```

This indicates **excellent experimental repeatability** across all thickness levels.

## Best Practices for Reporting

### When reporting regression results:
✅ **DO**:
- Report slope ± SE with 95% CI
- Include p-value for slope
- State R² and adjusted R²
- Show confidence bands on plots
- Explain what error bars represent

❌ **DON'T**:
- Report only R² without uncertainties
- Use "error bars overlap therefore not significant"
- Ignore multiple comparison issues
- Show error bars without defining them

### When comparing groups:
✅ **DO**:
- Use formal hypothesis tests (t-tests)
- Report mean difference with 95% CI
- Include p-values
- Apply Bonferroni correction when doing multiple comparisons
- Distinguish between statistical and practical significance

❌ **DON'T**:
- Rely solely on visual overlap of error bars
- Perform many tests without correction
- Confuse SD with SE
- Report "significant" without defining significance level

## References

1. **Welch's t-test**: Welch, B.L. (1947). "The generalization of 'Student's' problem when several different population variances are involved". Biometrika. 34 (1–2): 28–35.

2. **Bonferroni correction**: Dunn, O.J. (1961). "Multiple comparisons among means". Journal of the American Statistical Association. 56 (293): 52–64.

3. **Regression uncertainty**: Draper, N.R., Smith, H. (1998). Applied Regression Analysis (3rd ed.). Wiley.

4. **Error propagation**: Taylor, J.R. (1997). An Introduction to Error Analysis (2nd ed.). University Science Books.

## Software Implementation

All statistical methods are implemented in `statistics_advanced.py` using:
- **NumPy**: Numerical computations
- **SciPy**: Statistical distributions and tests (scipy.stats)
- **Pandas**: Data manipulation

Key functions:
- `weighted_linear_regression()`: Full regression with CIs
- `welch_t_test()`: Pairwise comparisons
- `adjacent_thickness_comparisons()`: Automated pairwise testing
- `compute_prediction_bands()`: Regression confidence/prediction bands
- `bonferroni_correction()`: Multiple comparison adjustment
