### Contact Duration Detection

Contact duration is the time period during which an impacting object remains in physical contact with the viscoelastic pad. Accurate detection of this interval is critical for understanding impact attenuation behavior.

#### Method 1: Force Threshold (Primary)

**Principle:** Contact is defined as the period when the magnitude of force exceeds a specified fraction of the peak force.

```python
contact_info = calculate_contact_duration_threshold(
    time=time_array,
    force=force_array,
    threshold_fraction=0.05,  # 5% threshold
    smooth=True
)
```

**Algorithm:**

1. Signal smoothing (optional):

   - Apply Savitzky-Golay filter to reduce noise
   - Window length: 11 points
   - Polynomial order: 3

2. Find maximum absolute force:
   $F_{\text{peak}} = \max(|F(t)|)$

3. Threshold calculation:

   - Threshold force: $F_{\text{threshold}} = \alpha F_{\text{peak}}$
   - Default: $\alpha = 0.05$

4. Contact start:

   - Search backward from peak
   - Start when $|F(t)| > F_{\text{threshold}}$
   - Time point: $t_{\text{start}}$

5. Contact end:

   - Search forward from peak
   - End when $|F(t)| < F_{\text{threshold}}$
   - Time point: $t_{\text{end}}$

6. Duration:
   $\Delta t = t_{\text{end}} - t_{\text{start}}$

**Advantages:**

- Robust to noise (with smoothing)
- Intuitive physical interpretation
- Works with force data only
- Validated in literature

**Limitations:**

- Threshold selection affects results
- May miss very brief contacts
- Sensitive to baseline drift

#### Method 2: Velocity-Based (Threshold Detection)

**Principle:** Contact occurs when velocity drops below a threshold, reaches minimum compression, then recovers. This method handles sparse velocity data with NaN values.

```python
contact_info = calculate_contact_duration_velocity(
    time=time_array,
    velocity=velocity_array,
    velocity_threshold=0.1,  # 0.1 m/s threshold
    smooth=True
)
```

**Algorithm:**

1. Data validation:
   - Filter out NaN values from sparse velocity data
   - Require minimum 10 valid measurements

2. Signal smoothing (optional):
   - Apply Savitzky-Golay filter if sufficient data points (≥11)
   - Window length: 11 points
   - Polynomial order: 3

3. Find minimum velocity:
   - $v_{\text{min}} = \min(v(t))$
   - Corresponds to maximum compression point

4. Contact start:
   - Search backward from $v_{\text{min}}$
   - Start when $v(t) > v_{\text{threshold}}$
   - Time point: $t_{\text{start}}$

5. Contact end:
   - Search forward from $v_{\text{min}}$
   - End when $|v(t)| > v_{\text{threshold}}$
   - Time point: $t_{\text{end}}$

6. Duration:
   $\Delta t = t_{\text{end}} - t_{\text{start}}$

**Advantages:**

- Handles sparse velocity data robustly
- Less sensitive to noise than zero-crossing
- Provides validation against force method

**Limitations:**

- Requires velocity measurements (derived from position)
- Threshold selection affects results
- May have gaps in data due to sensor limitations

#### Method 3: Energy-Based (Kinetic Energy Tracking)

**Principle:** Contact period corresponds to kinetic energy transformation during impact. Uses global maximum energy to identify impact event correctly.

```python
contact_info = calculate_contact_duration_energy(
    time=time_array,
    velocity=velocity_array,
    mass=0.5,  # cart mass in kg
    recovery_fraction=0.5,  # 50% energy recovery threshold
    smooth=True
)
```

**Algorithm:**

1. Data validation:
   - Filter out NaN values
   - Require minimum 10 valid measurements

2. Calculate kinetic energy:
   $E_k(t) = \tfrac{1}{2} m v^2(t)$

3. Find global maximum energy:
   - $E_{\text{initial}} = \max(E_k(t))$
   - Represents velocity before impact
   - Index: $t_{\text{max}}$

4. Find minimum energy after maximum:
   - $E_{\text{min}} = \min\{E_k(t) : t > t_{\text{max}}\}$
   - Maximum compression during impact
   - Time: $t_{\text{min}}$

5. Recovery threshold:
   - $E_{\text{threshold}} = \alpha \cdot E_{\text{initial}}$
   - Default: $\alpha = 0.5$ (50% recovery)

6. Contact start:
   - Search backward from $E_{\text{min}}$ to $E_{\text{max}}$
   - Start when $E_k(t) < E_{\text{threshold}}$
   - Time: $t_{\text{start}}$

7. Contact end:
   - Search forward from $E_{\text{min}}$
   - End when $E_k(t) \geq E_{\text{threshold}}$
   - Time: $t_{\text{end}}$

8. Duration:
   $\Delta t = t_{\text{end}} - t_{\text{start}}$

**Advantages:**

- Physical interpretation (energy conversion)
- Robust to velocity sign changes
- Finds correct impact event (not spurious low-energy regions)

**Limitations:**

- Requires mass parameter
- Depends on velocity measurements
- Recovery fraction selection affects results

### Linear Regression

#### Standard Linear Regression

Uses `scipy.stats.linregress` for least-squares fitting:

**Model:**
$y = mx + b$

**Outputs:**

- $m$: slope
- $b$: intercept
- $R^2$: coefficient of determination
- $p$: statistical significance
- $\mathrm{SE}$: standard error

#### Weighted Regression

When measurement uncertainties $\sigma_y$ are available:

**Weights:**
$w_i = \dfrac{1}{\sigma_{y_i}^2}$

**Minimizes:**
$\displaystyle \sum_i w_i (y_i - m x_i - b)^2$

### Uncertainty Estimation

#### Bootstrap Resampling

**Procedure:**

1. Perform $N_{\text{boot}} = 1000$ iterations

2. For each iteration:

   - Randomly sample data with replacement
   - Add noise: $y_{\text{boot}} = y + \mathcal{N}(0, \text{yerr})$
   - Compute $m_{\text{boot}}, b_{\text{boot}}$

3. Standard deviations:
   $\sigma_m = \operatorname{std}(m_{\text{boot}})$
   $\sigma_b = \operatorname{std}(b_{\text{boot}})$

#### Error Bar Method

**Minimum slope:**
$m_{\min} = \dfrac{(y_n - \sigma_n) - (y_1 + \sigma_1)}{x_n - x_1}$

**Maximum slope:**
$m_{\max} = \dfrac{(y_n + \sigma_n) - (y_1 - \sigma_1)}{x_n - x_1}$

**Uncertainty:**
$\sigma_m = \dfrac{m_{\max} - m_{\min}}{2}$

### Confidence Intervals

#### 95% Confidence Band

**Formula:**

$$
\mathrm{CI}(x) = t_{0.975,n-2} \cdot \mathrm{SE} \cdot
\sqrt{\frac{1}{n} + \frac{(x - \bar{x})^2}{S_{xx}}}
$$

Where:

- $\bar{x}$ is mean of $x$
- $S_{xx} = \sum (x_i - \bar{x})^2$

### Statistical Significance

**Null hypothesis:** $H_0: m = 0$
**Alternative:** $H_a: m \ne 0$

**Test statistic:**
$t = \dfrac{m}{\mathrm{SE}_m}$

**Decision:**

- $p < 0.05$: reject $H_0$
- $p \ge 0.05$: fail to reject

### Effect Size

Coefficient of determination:

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
$$

Where:
$SS_{\text{res}} = \sum (y_i - \hat{y}*i)^2$
$SS*{\text{tot}} = \sum (y_i - \bar{y})^2$

### Coefficient of Variation

Definition:

$$
\mathrm{CV} = \frac{\sigma}{\mu} \times 100%
$$

### Savitzky–Golay Filtering

No math changes besides variable formatting.

### Viscoelastic Theory

**Spring component:**
$F_{\text{elastic}} = k x$

**Dashpot component:**
$F_{\text{viscous}} = c v$

**Combined model:**
$F = kx + cv$

### Impulse–Momentum Theorem

$$
J = \int F, dt = \Delta p
$$

For fixed $\Delta p$:
Longer $\Delta t \Rightarrow$ smaller average $F$

### Energy Dissipation

**Initial kinetic energy:**
$E_0 = \tfrac{1}{2} m v_0^2$

**Coefficient of restitution:**
$e = \dfrac{v_{\text{rebound}}}{v_{\text{impact}}}$

**Energy loss:**
$E_{\text{loss}} = E_0(1 - e^2)$
