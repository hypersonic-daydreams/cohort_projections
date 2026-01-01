"""
COVID-19 Intervention Modeling
==============================

Creates intervention variables and estimates COVID-19 effects on the
international migration time series. The 2020 observation (30 migrants)
is a known outlier due to pandemic-related travel restrictions.

Per ADR-020, COVID is modeled as an intervention term rather than
excluded, preserving information about the exceptional year.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper


@dataclass
class CovidEffectResult:
    """Container for COVID intervention effect estimation."""

    model_with_covid: RegressionResultsWrapper
    model_without_covid: RegressionResultsWrapper
    covid_effect: float
    covid_effect_se: float
    covid_effect_pvalue: float
    aic_with: float
    aic_without: float
    bic_with: float
    bic_without: float
    preferred_model: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "covid_effect": self.covid_effect,
            "covid_effect_se": self.covid_effect_se,
            "covid_effect_pvalue": self.covid_effect_pvalue,
            "aic_with_covid": self.aic_with,
            "aic_without_covid": self.aic_without,
            "bic_with_covid": self.bic_with,
            "bic_without_covid": self.bic_without,
            "preferred_model": self.preferred_model,
            "covid_effect_significant": self.covid_effect_pvalue < 0.05,
        }


def create_covid_intervention(
    df: pd.DataFrame,
    year_col: str = "year",
) -> pd.DataFrame:
    """
    Create COVID-19 intervention indicator variables.

    Creates three types of intervention terms:
    1. Pulse: 1 for 2020 only (immediate shock)
    2. Step: 1 for 2020+ (permanent level shift)
    3. Recovery: Years since 2020 (recovery ramp)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a year column
    year_col : str
        Name of the year column

    Returns
    -------
    pd.DataFrame
        DataFrame with added COVID intervention columns
    """
    df = df.copy()

    # Pulse intervention (2020 only) - for modeling temporary shock
    df["covid_pulse"] = (df[year_col] == 2020).astype(int)

    # Step intervention (2020 onward) - for modeling sustained effects
    df["covid_step"] = (df[year_col] >= 2020).astype(int)

    # Recovery ramp (increasing from 2020) - for modeling gradual recovery
    df["covid_recovery"] = np.maximum(0, df[year_col] - 2020)

    # Combined post-COVID indicator excluding 2020 (for recovery analysis)
    df["post_covid_recovery"] = (df[year_col] > 2020).astype(int)

    return df


def estimate_covid_effect(
    df: pd.DataFrame,
    y_col: str,
    year_col: str = "year",
    intervention_type: str = "pulse",
    include_trend: bool = True,
    cov_type: str = "HAC",
    maxlags: int = 2,
) -> CovidEffectResult:
    """
    Estimate COVID-19 intervention effect by comparing models.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with year and dependent variable columns
    y_col : str
        Name of the dependent variable column
    year_col : str
        Name of the year column
    intervention_type : str
        Type of intervention: "pulse", "step", or "recovery"
    include_trend : bool
        Whether to include linear time trend
    cov_type : str
        Covariance type for standard errors
    maxlags : int
        Maximum lags for HAC standard errors

    Returns
    -------
    CovidEffectResult
        Results comparing models with and without COVID intervention
    """
    df = df.copy()
    df = create_covid_intervention(df, year_col)

    # Create time trend
    min_year = df[year_col].min()
    df["t"] = df[year_col] - min_year

    # Select intervention variable
    intervention_var = f"covid_{intervention_type}"
    if intervention_var not in df.columns:
        raise ValueError(f"Unknown intervention type: {intervention_type}")

    y = df[y_col]

    # Model without COVID intervention
    if include_trend:
        X_without = sm.add_constant(df[["t"]])
    else:
        X_without = sm.add_constant(pd.DataFrame({"_ones": np.ones(len(df))}))
        X_without = X_without.drop(columns=["_ones"])

    if cov_type == "HAC":
        model_without = OLS(y, X_without).fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags}
        )
    else:
        model_without = OLS(y, X_without).fit(cov_type=cov_type)

    # Model with COVID intervention
    if include_trend:
        X_with = sm.add_constant(df[["t", intervention_var]])
    else:
        X_with = sm.add_constant(df[[intervention_var]])

    if cov_type == "HAC":
        model_with = OLS(y, X_with).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    else:
        model_with = OLS(y, X_with).fit(cov_type=cov_type)

    # Extract COVID effect
    covid_effect = float(model_with.params[intervention_var])
    covid_effect_se = float(model_with.bse[intervention_var])
    covid_effect_pvalue = float(model_with.pvalues[intervention_var])

    # Model comparison
    aic_with = float(model_with.aic)
    aic_without = float(model_without.aic)
    bic_with = float(model_with.bic)
    bic_without = float(model_without.bic)

    # Preferred model based on BIC
    if bic_with < bic_without:
        preferred = "with_covid"
    else:
        preferred = "without_covid"

    return CovidEffectResult(
        model_with_covid=model_with,
        model_without_covid=model_without,
        covid_effect=covid_effect,
        covid_effect_se=covid_effect_se,
        covid_effect_pvalue=covid_effect_pvalue,
        aic_with=aic_with,
        aic_without=aic_without,
        bic_with=bic_with,
        bic_without=bic_without,
        preferred_model=preferred,
    )


def calculate_counterfactual_2020(
    df: pd.DataFrame,
    y_col: str,
    year_col: str = "year",
) -> dict:
    """
    Calculate counterfactual 2020 value based on trend from prior years.

    Uses 2015-2019 to estimate trend and project to 2020, then
    compares with actual 2020 value to quantify COVID impact.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with year and dependent variable columns
    y_col : str
        Name of the dependent variable column
    year_col : str
        Name of the year column

    Returns
    -------
    dict
        Counterfactual analysis results
    """
    # Subset to pre-COVID years for trend estimation
    df_pre = df[(df[year_col] >= 2015) & (df[year_col] <= 2019)].copy()

    if len(df_pre) < 3:
        return {"error": "Insufficient pre-COVID observations for trend estimation"}

    # Estimate trend from 2015-2019
    df_pre["t"] = df_pre[year_col] - 2015
    X = sm.add_constant(df_pre["t"])
    y = df_pre[y_col]
    model = OLS(y, X).fit()

    # Project to 2020 (t=5)
    counterfactual_2020 = float(model.params["const"] + model.params["t"] * 5)

    # Get actual 2020 value
    actual_2020 = df[df[year_col] == 2020][y_col].values
    if len(actual_2020) == 0:
        return {"error": "No 2020 observation found"}
    actual_2020 = float(actual_2020[0])

    # Calculate impact
    covid_impact = actual_2020 - counterfactual_2020
    covid_impact_pct = (
        (covid_impact / counterfactual_2020) * 100
        if counterfactual_2020 != 0
        else np.nan
    )

    return {
        "counterfactual_2020": counterfactual_2020,
        "actual_2020": actual_2020,
        "covid_impact": covid_impact,
        "covid_impact_pct": covid_impact_pct,
        "trend_slope": float(model.params["t"]),
        "trend_r_squared": float(model.rsquared),
        "baseline_years": "2015-2019",
    }
