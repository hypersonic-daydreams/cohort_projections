"""
Robust Inference for Regime-Specific Variance
==============================================

Provides heteroskedasticity-robust and regime-specific variance
estimation to account for the substantial variance heterogeneity
across vintages identified in Phase A (29:1 variance ratio).

Per ADR-020, robust standard errors are required for valid inference
given the variance differences across measurement regimes.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.linear_model import OLS, WLS, RegressionResultsWrapper


@dataclass
class RegimeVarianceResult:
    """Container for regime-specific variance analysis."""

    variances: dict[str, float]
    variance_ratio: float
    levene_statistic: float
    levene_pvalue: float
    bartlett_statistic: float
    bartlett_pvalue: float
    heteroskedastic: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "variances_by_regime": self.variances,
            "variance_ratio_max_min": self.variance_ratio,
            "levene_test": {
                "statistic": self.levene_statistic,
                "p_value": self.levene_pvalue,
            },
            "bartlett_test": {
                "statistic": self.bartlett_statistic,
                "p_value": self.bartlett_pvalue,
            },
            "heteroskedastic_at_05": self.heteroskedastic,
        }


@dataclass
class RobustSEComparison:
    """Container for comparing standard error estimators."""

    ols_se: dict[str, float]
    hc0_se: dict[str, float]
    hc1_se: dict[str, float]
    hc2_se: dict[str, float]
    hc3_se: dict[str, float]
    hac_se: dict[str, float]
    coefficients: dict[str, float]
    n_obs: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "coefficients": self.coefficients,
            "se_comparison": {
                "OLS": self.ols_se,
                "HC0": self.hc0_se,
                "HC1": self.hc1_se,
                "HC2": self.hc2_se,
                "HC3": self.hc3_se,
                "HAC": self.hac_se,
            },
            "n_obs": self.n_obs,
        }


def estimate_regime_variances(
    df: pd.DataFrame,
    y_col: str,
    regime_col: str = "vintage",
    detrend: bool = True,
) -> RegimeVarianceResult:
    """
    Estimate regime-specific error variances and test for heteroskedasticity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with dependent variable and regime indicator
    y_col : str
        Name of the dependent variable column
    regime_col : str
        Name of the regime/vintage column
    detrend : bool
        Whether to detrend within each regime before calculating variance

    Returns
    -------
    RegimeVarianceResult
        Variance estimates and heteroskedasticity tests
    """
    regimes = sorted(df[regime_col].unique())
    variances = {}
    residual_groups = []

    for regime in regimes:
        subset = df[df[regime_col] == regime].copy()
        y = subset[y_col].values

        if detrend and len(y) > 2:
            # Detrend within regime
            t = np.arange(len(y))
            X = sm.add_constant(t)
            try:
                residuals = OLS(y, X).fit().resid
            except Exception:
                residuals = y - np.mean(y)
        else:
            # Just demean
            residuals = y - np.mean(y)

        variances[str(regime)] = float(np.var(residuals, ddof=1))
        residual_groups.append(residuals)

    # Variance ratio (max/min)
    var_values = [v for v in variances.values() if v > 0]
    variance_ratio = (
        max(var_values) / min(var_values) if min(var_values) > 0 else np.inf
    )

    # Levene's test (robust to non-normality)
    levene_stat, levene_p = stats.levene(*residual_groups)

    # Bartlett's test (assumes normality)
    try:
        bartlett_stat, bartlett_p = stats.bartlett(*residual_groups)
    except Exception:
        bartlett_stat, bartlett_p = np.nan, np.nan

    return RegimeVarianceResult(
        variances=variances,
        variance_ratio=float(variance_ratio),
        levene_statistic=float(levene_stat),
        levene_pvalue=float(levene_p),
        bartlett_statistic=float(bartlett_stat),
        bartlett_pvalue=float(bartlett_p),
        heteroskedastic=bool(levene_p < 0.05),
    )


def estimate_with_robust_se(
    df: pd.DataFrame,
    y_col: str,
    X_cols: list[str],
    maxlags: int = 2,
) -> RobustSEComparison:
    """
    Estimate model with multiple robust standard error specifications.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with dependent and independent variables
    y_col : str
        Name of the dependent variable column
    X_cols : list[str]
        Names of independent variable columns
    maxlags : int
        Maximum lags for HAC standard errors

    Returns
    -------
    RobustSEComparison
        Comparison of SE estimates across methods
    """
    X = sm.add_constant(df[X_cols])
    y = df[y_col]

    # Fit with different covariance estimators
    model_ols = OLS(y, X).fit()
    model_hc0 = OLS(y, X).fit(cov_type="HC0")
    model_hc1 = OLS(y, X).fit(cov_type="HC1")
    model_hc2 = OLS(y, X).fit(cov_type="HC2")
    model_hc3 = OLS(y, X).fit(cov_type="HC3")
    model_hac = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

    # Extract standard errors
    all_vars = ["const"] + X_cols

    def extract_se(model: RegressionResultsWrapper) -> dict[str, float]:
        return {v: float(model.bse[v]) for v in all_vars if v in model.bse.index}

    return RobustSEComparison(
        ols_se=extract_se(model_ols),
        hc0_se=extract_se(model_hc0),
        hc1_se=extract_se(model_hc1),
        hc2_se=extract_se(model_hc2),
        hc3_se=extract_se(model_hc3),
        hac_se=extract_se(model_hac),
        coefficients={
            v: float(model_ols.params[v])
            for v in all_vars
            if v in model_ols.params.index
        },
        n_obs=int(model_ols.nobs),
    )


def estimate_wls_by_regime(
    df: pd.DataFrame,
    y_col: str,
    X_cols: list[str],
    regime_col: str = "vintage",
) -> dict:
    """
    Estimate Weighted Least Squares with weights inverse to regime variance.

    This corrects for heteroskedasticity by downweighting high-variance
    regimes (typically the 2020s with extreme values).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with dependent and independent variables
    y_col : str
        Name of the dependent variable column
    X_cols : list[str]
        Names of independent variable columns
    regime_col : str
        Name of the regime/vintage column

    Returns
    -------
    dict
        WLS results including weights used
    """
    # First estimate regime variances
    var_result = estimate_regime_variances(df, y_col, regime_col)

    # Create weights (inverse variance)
    df = df.copy()
    df["_weight"] = (
        df[regime_col].astype(str).map(lambda r: 1.0 / var_result.variances.get(r, 1.0))
    )

    # Normalize weights
    df["_weight"] = df["_weight"] / df["_weight"].mean()

    X = sm.add_constant(df[X_cols])
    y = df[y_col]

    # WLS estimation
    model = WLS(y, X, weights=df["_weight"]).fit()

    # Also run OLS for comparison
    model_ols = OLS(y, X).fit()

    return {
        "wls_params": model.params.to_dict(),
        "wls_bse": model.bse.to_dict(),
        "wls_pvalues": model.pvalues.to_dict(),
        "wls_rsquared": float(model.rsquared),
        "ols_params": model_ols.params.to_dict(),
        "ols_bse": model_ols.bse.to_dict(),
        "regime_weights": {r: 1.0 / v for r, v in var_result.variances.items()},
        "regime_variances": var_result.variances,
        "n_obs": int(model.nobs),
    }
