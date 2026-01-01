"""
Piecewise Trend Estimation
==========================

Estimates regime-specific linear trends, allowing slope to differ
across vintage periods. This helps distinguish genuine trend changes
from methodology artifacts.

Per ADR-020, trend estimation uses HAC standard errors to account
for autocorrelation in the time series.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper

from .vintage_dummies import create_regime_dummies


@dataclass
class PiecewiseTrendResult:
    """Container for piecewise trend estimation results."""

    model: RegressionResultsWrapper
    slopes: dict[str, float]
    slope_se: dict[str, float]
    intercepts: dict[str, float]
    slope_equality_test: Optional[dict] = None
    r_squared: float = 0.0
    n_obs: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "slopes": self.slopes,
            "slope_se": self.slope_se,
            "intercepts": self.intercepts,
            "slope_equality_test": self.slope_equality_test,
            "r_squared": self.r_squared,
            "n_obs": self.n_obs,
        }


def estimate_piecewise_trend(
    df: pd.DataFrame,
    y_col: str,
    year_col: str = "year",
    cov_type: str = "HAC",
    maxlags: int = 2,
) -> PiecewiseTrendResult:
    """
    Estimate piecewise linear trend with regime-specific slopes.

    The model is:
        y_t = alpha + beta_1*regime_2010s + beta_2*regime_2020s +
              gamma_1*trend_2000s + gamma_2*trend_2010s + gamma_3*trend_2020s + eps

    Time trends are reset at each regime boundary for cleaner interpretation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with year and dependent variable columns
    y_col : str
        Name of the dependent variable column
    year_col : str
        Name of the year column
    cov_type : str
        Covariance type: "HAC" (default), "HC3", or "nonrobust"
    maxlags : int
        Maximum lags for HAC standard errors

    Returns
    -------
    PiecewiseTrendResult
        Results container with slope estimates and tests
    """
    df = df.copy()
    df = create_regime_dummies(df, year_col)

    # Create centered time variable (t=0 at first observation)
    min_year = df[year_col].min()
    df["t"] = df[year_col] - min_year

    # Regime-specific trend interactions (reset at each boundary)
    # 2000s: t runs from 0 to 9
    df["trend_2000s"] = df["t"] * df["regime_2000s"]

    # 2010s: t resets, runs from 0 to 9
    df["trend_2010s"] = (df[year_col] - 2010) * df["regime_2010s"]
    df["trend_2010s"] = df["trend_2010s"].clip(lower=0)

    # 2020s: t resets, runs from 0 to 4
    df["trend_2020s"] = (df[year_col] - 2020) * df["regime_2020s"]
    df["trend_2020s"] = df["trend_2020s"].clip(lower=0)

    # Design matrix: level shifts + regime-specific trends
    X_cols = [
        "regime_2010s",
        "regime_2020s",
        "trend_2000s",
        "trend_2010s",
        "trend_2020s",
    ]
    X = sm.add_constant(df[X_cols])
    y = df[y_col]

    # Estimate with specified covariance type
    if cov_type == "HAC":
        model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    elif cov_type == "HC3":
        model = OLS(y, X).fit(cov_type="HC3")
    else:
        model = OLS(y, X).fit()

    # Extract slope coefficients
    slopes = {
        "2000s": float(model.params["trend_2000s"]),
        "2010s": float(model.params["trend_2010s"]),
        "2020s": float(model.params["trend_2020s"]),
    }

    slope_se = {
        "2000s": float(model.bse["trend_2000s"]),
        "2010s": float(model.bse["trend_2010s"]),
        "2020s": float(model.bse["trend_2020s"]),
    }

    # Extract level shift coefficients (intercept adjustments)
    intercepts = {
        "2000s_base": float(model.params["const"]),
        "2010s_shift": float(model.params["regime_2010s"]),
        "2020s_shift": float(model.params["regime_2020s"]),
    }

    # Test for slope equality across regimes
    try:
        f_test = model.f_test("trend_2000s = trend_2010s = trend_2020s")
        slope_equality_test = {
            "f_statistic": float(f_test.fvalue),
            "p_value": float(f_test.pvalue),
            "df_num": int(f_test.df_num),
            "df_denom": float(f_test.df_denom),
            "reject_equality_at_05": bool(f_test.pvalue < 0.05),
        }
    except Exception:
        # F-test may fail with small samples
        slope_equality_test = None

    return PiecewiseTrendResult(
        model=model,
        slopes=slopes,
        slope_se=slope_se,
        intercepts=intercepts,
        slope_equality_test=slope_equality_test,
        r_squared=float(model.rsquared),
        n_obs=int(model.nobs),
    )


def estimate_simple_trend(
    df: pd.DataFrame,
    y_col: str,
    year_col: str = "year",
    cov_type: str = "HAC",
    maxlags: int = 2,
) -> dict:
    """
    Estimate a simple linear trend (single slope for entire series).

    This serves as a baseline comparison for the piecewise model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with year and dependent variable columns
    y_col : str
        Name of the dependent variable column
    year_col : str
        Name of the year column
    cov_type : str
        Covariance type
    maxlags : int
        Maximum lags for HAC

    Returns
    -------
    dict
        Simple trend results
    """
    df = df.copy()
    min_year = df[year_col].min()
    df["t"] = df[year_col] - min_year

    X = sm.add_constant(df["t"])
    y = df[y_col]

    if cov_type == "HAC":
        model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    elif cov_type == "HC3":
        model = OLS(y, X).fit(cov_type="HC3")
    else:
        model = OLS(y, X).fit()

    return {
        "slope": float(model.params["t"]),
        "slope_se": float(model.bse["t"]),
        "slope_pvalue": float(model.pvalues["t"]),
        "intercept": float(model.params["const"]),
        "r_squared": float(model.rsquared),
        "n_obs": int(model.nobs),
    }
