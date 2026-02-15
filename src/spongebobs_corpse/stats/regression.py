"""

Advanced Statistical Analysis Module.

This module provides rigorous statistical methods beyond simple error-bar overlap:

1. Regression with parameter uncertainties (confidence intervals on slope/intercept)

2. Hypothesis tests for pairwise comparisons (adjacent thickness levels)

3. Clear definitions of uncertainty types (random, systematic, combined)

4. Goodness-of-fit tests and residual analysis

Error Bar Definitions:

- Standard Error (SE): std / sqrt(n) - represents uncertainty in the mean

- Standard Deviation (SD): spread of individual measurements

- 95% Confidence Interval: mean +/- t * SE

- Combined Uncertainty: sqrt(u_random^2 + u_instrument^2)

"""

import numpy as np

import pandas as pd

from scipy import stats

from dataclasses import dataclass

from typing import Dict, List, Tuple, Optional


@dataclass
class RegressionResults:
    """Container for regression analysis results with uncertainties."""

    slope: float

    slope_se: float
    slope_ci_lower: float
    slope_ci_upper: float
    intercept: float

    intercept_se: float
    intercept_ci_lower: float
    intercept_ci_upper: float
    r_squared: float

    adj_r_squared: float
    rmse: float
    p_value_slope: float
    p_value_intercept: float
    n_points: int

    degrees_of_freedom: int

    residuals: np.ndarray

    fitted_values: np.ndarray

    def summary_text(self, x_name: str = "x", y_name: str = "y") -> str:
        """Generate a formatted summary of regression results."""

        text = f"Linear Regression: {y_name} = slope * {x_name} + intercept\n"

        text += f"=" * 70 + "\n"

        text += f"Slope:     {self.slope:.6f} ± {self.slope_se:.6f}\n"

        text += f"  95% CI:  [{self.slope_ci_lower:.6f}, {self.slope_ci_upper:.6f}]\n"

        text += f"  p-value: {self.p_value_slope:.4e}\n"

        text += f"\n"

        text += f"Intercept: {self.intercept:.6f} ± {self.intercept_se:.6f}\n"

        text += f"  95% CI:  [{self.intercept_ci_lower:.6f}, {self.intercept_ci_upper:.6f}]\n"

        text += f"  p-value: {self.p_value_intercept:.4e}\n"

        text += f"\n"

        text += f"Goodness of Fit:\n"

        text += f"  R²:      {self.r_squared:.4f}\n"

        text += f"  Adj. R²: {self.adj_r_squared:.4f}\n"

        text += f"  RMSE:    {self.rmse:.6f}\n"

        text += f"  N:       {self.n_points}\n"

        text += f"  DOF:     {self.degrees_of_freedom}\n"

        return text


@dataclass
class PairwiseComparison:
    """Container for pairwise comparison results."""

    group1: str

    group2: str

    mean1: float

    mean2: float

    se1: float
    se2: float
    mean_diff: float
    se_diff: float
    t_statistic: float

    p_value: float

    degrees_of_freedom: float

    ci_lower: float
    ci_upper: float

    significant: bool

    def summary_text(self) -> str:
        """Generate formatted summary of pairwise comparison."""

        sig_marker = (
            "***"
            if self.p_value < 0.001
            else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else "ns"
        )

        text = f"{self.group1} vs {self.group2}:\n"

        text += f"  Mean difference: {self.mean_diff:.6f} ± {self.se_diff:.6f}\n"

        text += f"  95% CI: [{self.ci_lower:.6f}, {self.ci_upper:.6f}]\n"

        text += f"  t({self.degrees_of_freedom:.1f}) = {self.t_statistic:.3f}, p = {self.p_value:.4f} {sig_marker}\n"

        return text


def analyze_residuals(residuals: np.ndarray) -> str:
    """

    Analyze residuals for systematic patterns using the Runs Test.

    Args:

        residuals: Array of residuals from regression

    Returns:

        String description of residual patterns

    """

    res = residuals[residuals != 0]

    n = len(res)

    if n < 4:

        return "Insufficient data for residual analysis"

    signs = np.sign(res)

    runs = 1

    for i in range(1, n):

        if signs[i] != signs[i - 1]:

            runs += 1

    n_pos = np.sum(signs > 0)

    n_neg = np.sum(signs < 0)

    if n_pos == 0 or n_neg == 0:

        return "Systematic bias (all residuals have same sign)"

    expected_runs = 1 + (2 * n_pos * n_neg) / n

    std_runs = np.sqrt((2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1)))

    z_score = (runs - expected_runs) / std_runs if std_runs > 0 else 0

    if z_score < -1.5:
        return "Systematic curvature suggested (residuals clustered)"

    elif z_score > 1.96:

        return "Rapid oscillation detected"

    else:

        return "Residuals appear randomly scattered"


def weighted_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> RegressionResults:
    """

    Perform weighted linear regression with full uncertainty quantification.

    Args:

        x: Independent variable

        y: Dependent variable

        weights: Optional weights (1/sigma^2 for weighted least squares)

        alpha: Significance level for confidence intervals (default 0.05 for 95% CI)

    Returns:

        RegressionResults object with all statistics and uncertainties

    """

    mask = np.isfinite(x) & np.isfinite(y)

    if weights is not None:

        mask &= np.isfinite(weights) & (weights > 0)

    x = x[mask]

    y = y[mask]

    if weights is not None:

        weights = weights[mask]

    n = len(x)

    if n < 3:

        raise ValueError("Need at least 3 data points for regression")

    if weights is None:

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        y_pred = slope * x + intercept

        residuals = y - y_pred

        rss = np.sum(residuals**2)

        tss = np.sum((y - np.mean(y)) ** 2)

        dof = n - 2

        rmse = np.sqrt(rss / dof)

        x_mean = np.mean(x)

        sxx = np.sum((x - x_mean) ** 2)

        se_slope = rmse / np.sqrt(sxx)

        se_intercept = rmse * np.sqrt(1 / n + x_mean**2 / sxx)

        t_crit = stats.t.ppf(1 - alpha / 2, dof)

        slope_ci_lower = slope - t_crit * se_slope

        slope_ci_upper = slope + t_crit * se_slope

        intercept_ci_lower = intercept - t_crit * se_intercept

        intercept_ci_upper = intercept + t_crit * se_intercept

        p_value_slope = p_value

        t_intercept = intercept / se_intercept

        p_value_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), dof))

        r_squared = r_value**2

    else:

        W = np.diag(weights)

        X = np.column_stack([np.ones(n), x])

        XtWX = X.T @ W @ X

        XtWy = X.T @ W @ y

        params = np.linalg.solve(XtWX, XtWy)

        intercept, slope = params

        y_pred = slope * x + intercept

        residuals = y - y_pred

        weighted_rss = np.sum(weights * residuals**2)

        dof = n - 2

        rmse = np.sqrt(np.sum(residuals**2) / dof)

        try:

            cov_matrix = np.linalg.inv(XtWX)

        except np.linalg.LinAlgError:

            return weighted_linear_regression(x, y, weights=None, alpha=alpha)

        se_intercept = np.sqrt(cov_matrix[0, 0])

        se_slope = np.sqrt(cov_matrix[1, 1])

        t_crit = stats.t.ppf(1 - alpha / 2, dof)

        slope_ci_lower = slope - t_crit * se_slope

        slope_ci_upper = slope + t_crit * se_slope

        intercept_ci_lower = intercept - t_crit * se_intercept

        intercept_ci_upper = intercept + t_crit * se_intercept

        t_slope = slope / se_slope

        p_value_slope = 2 * (1 - stats.t.cdf(abs(t_slope), dof))

        t_intercept = intercept / se_intercept

        p_value_intercept = 2 * (1 - stats.t.cdf(abs(t_intercept), dof))

        y_mean = np.average(y, weights=weights)

        ss_tot = np.sum(weights * (y - y_mean) ** 2)

        ss_res = weighted_rss

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / dof if dof > 0 else 0

    return RegressionResults(
        slope=slope,
        slope_se=se_slope,
        slope_ci_lower=slope_ci_lower,
        slope_ci_upper=slope_ci_upper,
        intercept=intercept,
        intercept_se=se_intercept,
        intercept_ci_lower=intercept_ci_lower,
        intercept_ci_upper=intercept_ci_upper,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        rmse=rmse,
        p_value_slope=p_value_slope,
        p_value_intercept=p_value_intercept,
        n_points=n,
        degrees_of_freedom=dof,
        residuals=residuals,
        fitted_values=y_pred,
    )


def fit_linear(
    x: np.ndarray,
    y: np.ndarray,
    yerr: Optional[np.ndarray] = None,
    method: str = "OLS",
) -> Dict:
    """
    Fit a linear model y = m*x + c using OLS or WLS.

    Args:
        x: Independent variable
        y: Dependent variable
        yerr: Optional y-uncertainty array
        method: "OLS" or "WLS"

    Returns:
        Dictionary with regression object and flattened fit stats.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    yerr_clean = None
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        mask &= np.isfinite(yerr)
        yerr_clean = yerr[mask]

    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        raise ValueError("Need at least 3 valid points for linear fitting")

    method_norm = method.upper()
    weights = None
    method_used = "OLS"
    if method_norm == "WLS" and yerr_clean is not None:
        valid = yerr_clean > 0
        if np.any(valid):
            x_clean = x_clean[valid]
            y_clean = y_clean[valid]
            yerr_clean = yerr_clean[valid]
            weights = 1.0 / np.square(yerr_clean)
            method_used = "WLS"

    regression = weighted_linear_regression(x_clean, y_clean, weights=weights)
    return {
        "model": "linear",
        "equation": "y = m*x + c",
        "method": method_used,
        "x": x_clean,
        "y": y_clean,
        "yerr": yerr_clean,
        "regression": regression,
        "slope": regression.slope,
        "intercept": regression.intercept,
        "slope_se": regression.slope_se,
        "intercept_se": regression.intercept_se,
        "slope_ci_lower": regression.slope_ci_lower,
        "slope_ci_upper": regression.slope_ci_upper,
        "intercept_ci_lower": regression.intercept_ci_lower,
        "intercept_ci_upper": regression.intercept_ci_upper,
        "r2": regression.r_squared,
        "adj_r2": regression.adj_r_squared,
        "p_value_slope": regression.p_value_slope,
        "p_value_intercept": regression.p_value_intercept,
        "n_points": regression.n_points,
        "fitted_values": regression.fitted_values,
        "residuals": regression.residuals,
    }


def fit_powerlaw(
    x: np.ndarray,
    y: np.ndarray,
    yerr: Optional[np.ndarray] = None,
    method: str = "OLS",
) -> Dict:
    """
    Fit the IA power-law surrogate y = a + b*sqrt(x).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if yerr is not None:
        mask &= np.isfinite(np.asarray(yerr, dtype=float))
    x_valid = x[mask]
    if np.any(x_valid <= 0):
        raise ValueError("Power-law fit requires x > 0 for sqrt(x)")

    x_transformed = np.sqrt(x_valid)
    y_valid = y[mask]
    yerr_valid = np.asarray(yerr, dtype=float)[mask] if yerr is not None else None
    result = fit_linear(x_transformed, y_valid, yerr=yerr_valid, method=method)
    result["model"] = "powerlaw_sqrt"
    result["equation"] = "y = a + b*sqrt(x)"
    result["x_original"] = x_valid
    result["x_transform"] = "sqrt(x)"
    return result


def fit_exponential(
    x: np.ndarray,
    y: np.ndarray,
    yerr: Optional[np.ndarray] = None,
    method: str = "OLS",
) -> Dict:
    """
    Fit exponential model y = a*exp(b*x) by linearizing ln(y).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)

    yerr_log = None
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=float)
        mask &= np.isfinite(yerr) & (yerr > 0)
        yerr_log = yerr[mask] / y[mask]

    x_masked = x[mask]
    y_masked = y[mask]
    y_log = np.log(y_masked)

    lin_result = fit_linear(x_masked, y_log, yerr=yerr_log, method=method)
    a = float(np.exp(lin_result["intercept"]))
    b = float(lin_result["slope"])
    a_se = abs(a * lin_result["intercept_se"])

    lin_result["model"] = "exponential"
    lin_result["equation"] = "y = a*exp(b*x)"
    lin_result["a"] = a
    lin_result["a_se"] = a_se
    lin_result["b"] = b
    lin_result["b_se"] = lin_result["slope_se"]
    return lin_result


def confidence_band_linear(
    x_grid: np.ndarray,
    results: Dict,
    level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence band of the mean response for a fitted linear-like model.
    """
    if "regression" not in results or "x" not in results:
        raise KeyError("results must come from fit_linear/fit_powerlaw/fit_exponential")

    alpha = 1.0 - level
    _, ci_lower, ci_upper, _, _ = compute_prediction_bands(
        np.asarray(results["x"], dtype=float),
        results["regression"],
        np.asarray(x_grid, dtype=float),
        alpha=alpha,
    )
    return ci_lower, ci_upper


def welch_t_test(
    mean1: float,
    std1: float,
    n1: int,
    mean2: float,
    std2: float,
    n2: int,
    alpha: float = 0.05,
) -> PairwiseComparison:
    """

    Perform Welch's t-test for comparing two means with potentially unequal variances.

    This is more robust than Student's t-test when variances differ.

    Args:

        mean1, std1, n1: Mean, standard deviation, and sample size for group 1

        mean2, std2, n2: Mean, standard deviation, and sample size for group 2

        alpha: Significance level (default 0.05)

    Returns:

        PairwiseComparison object with test results

    """

    se1 = std1 / np.sqrt(n1)

    se2 = std2 / np.sqrt(n2)

    mean_diff = mean1 - mean2

    se_diff = np.sqrt(se1**2 + se2**2)

    t_stat = mean_diff / se_diff if se_diff > 0 else 0

    if se1 > 0 and se2 > 0:

        dof = (se1**2 + se2**2) ** 2 / (se1**4 / (n1 - 1) + se2**4 / (n2 - 1))

    else:

        dof = n1 + n2 - 2

    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))

    t_crit = stats.t.ppf(1 - alpha / 2, dof)

    ci_lower = mean_diff - t_crit * se_diff

    ci_upper = mean_diff + t_crit * se_diff

    significant = p_value < alpha

    return PairwiseComparison(
        group1="Group 1",
        group2="Group 2",
        mean1=mean1,
        mean2=mean2,
        se1=se1,
        se2=se2,
        mean_diff=mean_diff,
        se_diff=se_diff,
        t_statistic=t_stat,
        p_value=p_value,
        degrees_of_freedom=dof,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant=significant,
    )


def adjacent_thickness_comparisons(
    summary_df: pd.DataFrame,
    value_col: str = "duration_s_mean",
    std_col: str = "duration_s_std",
    count_col: str = "duration_s_count",
    thickness_col: str = "thickness_mm",
    alpha: float = 0.05,
) -> List[PairwiseComparison]:
    """

    Perform pairwise comparisons between adjacent thickness levels.

    Args:

        summary_df: DataFrame with aggregated results by thickness

        value_col: Column name for mean values

        std_col: Column name for standard deviations

        count_col: Column name for sample sizes

        thickness_col: Column name for thickness values

        alpha: Significance level

    Returns:

        List of PairwiseComparison objects

    """

    df_sorted = summary_df.sort_values(thickness_col).reset_index(drop=True)

    comparisons = []

    for i in range(len(df_sorted) - 1):

        row1 = df_sorted.iloc[i]

        row2 = df_sorted.iloc[i + 1]

        thickness1 = row1[thickness_col]

        thickness2 = row2[thickness_col]

        mean1 = row1[value_col]

        mean2 = row2[value_col]

        std1 = row1[std_col]

        std2 = row2[std_col]

        n1 = int(row1[count_col])

        n2 = int(row2[count_col])

        comparison = welch_t_test(mean1, std1, n1, mean2, std2, n2, alpha)

        comparison.group1 = f"{thickness1:.1f} mm"

        comparison.group2 = f"{thickness2:.1f} mm"

        comparisons.append(comparison)

    return comparisons


def compute_prediction_bands(
    x: np.ndarray,
    reg_results: RegressionResults,
    x_pred: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    Compute confidence and prediction bands for regression.

    Args:

        x: Original x data used for regression

        reg_results: RegressionResults object

        x_pred: x values for prediction

        alpha: Significance level (default 0.05 for 95% bands)

    Returns:

        Tuple of (y_pred, ci_lower, ci_upper, pi_lower, pi_upper)

        - y_pred: Predicted y values

        - ci_lower, ci_upper: 95% confidence interval for the mean

        - pi_lower, pi_upper: 95% prediction interval for individual observations

    """

    y_pred = reg_results.slope * x_pred + reg_results.intercept

    n = reg_results.n_points

    dof = reg_results.degrees_of_freedom

    x_mean = np.mean(x)

    sxx = np.sum((x - x_mean) ** 2)

    t_crit = stats.t.ppf(1 - alpha / 2, dof)

    se_ci = reg_results.rmse * np.sqrt(1 / n + (x_pred - x_mean) ** 2 / sxx)

    ci_lower = y_pred - t_crit * se_ci

    ci_upper = y_pred + t_crit * se_ci

    se_pi = reg_results.rmse * np.sqrt(1 + 1 / n + (x_pred - x_mean) ** 2 / sxx)

    pi_lower = y_pred - t_crit * se_pi

    pi_upper = y_pred + t_crit * se_pi

    return y_pred, ci_lower, ci_upper, pi_lower, pi_upper


def residual_analysis(reg_results: RegressionResults, x: np.ndarray) -> Dict:
    """

    Perform residual diagnostics for regression.

    Tests for:

    - Normality (Shapiro-Wilk test)

    - Homoscedasticity (Breusch-Pagan test approximation)

    - Autocorrelation (Durbin-Watson statistic)

    Args:

        reg_results: RegressionResults object

        x: Independent variable values

    Returns:

        Dictionary with diagnostic test results

    """

    residuals = reg_results.residuals

    n = len(residuals)

    diagnostics = {}

    if n >= 3:

        shapiro_stat, shapiro_p = stats.shapiro(residuals)

        diagnostics["normality_test"] = {
            "test": "Shapiro-Wilk",
            "statistic": shapiro_stat,
            "p_value": shapiro_p,
            "normal": shapiro_p > 0.05,
        }

    std_residuals = residuals / reg_results.rmse

    diagnostics["std_residuals"] = std_residuals

    outliers = np.abs(std_residuals) > 3

    diagnostics["n_outliers"] = np.sum(outliers)

    diagnostics["outlier_indices"] = np.where(outliers)[0]

    if n > 1:

        dw = np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2)

        diagnostics["durbin_watson"] = {
            "statistic": dw,
            "interpretation": (
                "no autocorrelation" if 1.5 < dw < 2.5 else "possible autocorrelation"
            ),
        }

    if n > 2:

        corr, p_val = stats.pearsonr(np.abs(residuals), x)

        diagnostics["heteroscedasticity"] = {
            "correlation": corr,
            "p_value": p_val,
            "homoscedastic": p_val > 0.05,
        }

    return diagnostics


def error_bar_type_annotation(
    error_type: str = "se", n: int = None, confidence_level: float = 0.95
) -> str:
    """

    Generate clear annotation text explaining what error bars represent.

    Args:

        error_type: Type of error bars ("se", "sd", "ci", "uncertainty")

        n: Sample size (for SE and CI)

        confidence_level: Confidence level for CI (default 0.95)

    Returns:

        Descriptive text for error bars

    """

    if error_type == "se":

        if n:

            return f"Error bars: Standard Error (SD/√{n})"

        else:

            return "Error bars: Standard Error (SD/√n)"

    elif error_type == "sd":

        return "Error bars: Standard Deviation (spread of measurements)"

    elif error_type == "ci":

        pct = int(confidence_level * 100)

        return f"Error bars: {pct}% Confidence Interval"

    elif error_type == "uncertainty":

        return "Error bars: Combined Uncertainty (√(u_random² + u_instrument²))"

    else:

        return "Error bars: Uncertainty"


def format_p_value(p: float) -> str:
    """Format p-value with appropriate precision and notation."""

    if p < 0.001:

        return "p < 0.001***"

    elif p < 0.01:

        return f"p = {p:.3f}**"

    elif p < 0.05:

        return f"p = {p:.3f}*"

    else:

        return f"p = {p:.3f} (ns)"


def bonferroni_correction(
    p_values: List[float], alpha: float = 0.05
) -> Tuple[List[bool], float]:
    """

    Apply Bonferroni correction for multiple comparisons.

    Args:

        p_values: List of p-values from multiple tests

        alpha: Family-wise error rate (default 0.05)

    Returns:

        Tuple of (list of significant flags, adjusted alpha)

    """

    n_tests = len(p_values)

    alpha_adjusted = alpha / n_tests

    significant = [p < alpha_adjusted for p in p_values]

    return significant, alpha_adjusted
