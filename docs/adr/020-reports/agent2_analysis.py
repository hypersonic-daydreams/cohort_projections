#!/usr/bin/env python3
"""
Agent 2: Statistical Transition Analysis for ADR-019
Investigates vintage-related artifacts in North Dakota international migration time series.

Author: Agent 2 (Claude Opus 4.5)
Date: 2026-01-01

.. deprecated:: 2026-01-01
    This is a **legacy Phase A research script** from the ADR-019/020 investigation.
    It was used for one-time exploratory analysis and is retained for reproducibility
    and audit purposes only. This script is NOT production code and should NOT be
    modified or extended.

    The analysis outputs from this script have been incorporated into the final
    ADR-020 decision. For current methodology, see:
    - sdc_2024_replication/scripts/statistical_analysis/module_B1_regime_aware_models.py
    - sdc_2024_replication/scripts/statistical_analysis/module_B2_multistate_placebo.py

Status: DEPRECATED / ARCHIVED
Linting: Exempted from strict linting (see pyproject.toml per-file-ignores)
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bartlett, levene, mannwhitneyu, ttest_ind
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")

# Configuration
OUTPUT_DIR = Path("/home/nhaarstad/workspace/demography/cohort_projections/docs/adr/020-reports")
DATA_PATH = Path(
    "/home/nhaarstad/workspace/demography/cohort_projections/data/processed/immigration/state_migration_components_2000_2024.csv"
)

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_nd_data():
    """Load North Dakota international migration data."""
    df = pd.read_csv(DATA_PATH)
    nd_df = df[df["STATE"] == 38].copy()
    nd_df = nd_df.sort_values("year").reset_index(drop=True)

    # Create vintage period labels
    def get_vintage_period(row):
        if row["vintage"] == 2009:
            return "Vintage 2009 (2000-2009)"
        elif row["vintage"] == 2020:
            return "Vintage 2020 (2010-2019)"
        else:
            return "Vintage 2024 (2020-2024)"

    nd_df["vintage_period"] = nd_df.apply(get_vintage_period, axis=1)

    return nd_df


def compute_effect_size_cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def level_shift_analysis(nd_df):
    """
    Test for level shifts at vintage transition points.
    Returns detailed results for each transition.
    """
    results = []

    # Extract data by vintage
    v2009 = nd_df[nd_df["vintage"] == 2009]["INTERNATIONALMIG"].values
    v2020 = nd_df[nd_df["vintage"] == 2020]["INTERNATIONALMIG"].values
    v2024 = nd_df[nd_df["vintage"] == 2024]["INTERNATIONALMIG"].values

    # --- Transition 1: 2009 -> 2010 (Vintage 2009 to Vintage 2020) ---
    # Two-sample t-test (assumes equal variance)
    t_stat_1, p_val_t_1 = ttest_ind(v2009, v2020)
    # Welch's t-test (does not assume equal variance)
    t_stat_w1, p_val_w1 = ttest_ind(v2009, v2020, equal_var=False)
    # Mann-Whitney U test (non-parametric)
    u_stat_1, p_val_mw1 = mannwhitneyu(v2009, v2020, alternative="two-sided")

    # Effect size
    cohens_d_1 = compute_effect_size_cohens_d(v2009, v2020)

    # Basic statistics
    mean_v2009 = np.mean(v2009)
    mean_v2020 = np.mean(v2020)
    diff_1 = mean_v2020 - mean_v2009

    results.append(
        {
            "test_id": "LS-T1-TTEST",
            "test_name": "Two-sample t-test",
            "transition": "2009→2010 (Vintage 2009 vs Vintage 2020)",
            "hypothesis": "H0: No difference in mean international migration between Vintage 2009 and Vintage 2020",
            "test_statistic": t_stat_1,
            "statistic_name": "t-statistic",
            "df": len(v2009) + len(v2020) - 2,
            "p_value": p_val_t_1,
            "significance_level": 0.05,
            "reject_null": p_val_t_1 < 0.05,
            "effect_size": cohens_d_1,
            "effect_size_name": "Cohen's d",
            "interpretation": "Significant" if p_val_t_1 < 0.05 else "Not significant",
            "notes": f"Mean V2009={mean_v2009:.1f}, Mean V2020={mean_v2020:.1f}, Diff={diff_1:.1f}",
        }
    )

    results.append(
        {
            "test_id": "LS-T1-WELCH",
            "test_name": "Welch's t-test",
            "transition": "2009→2010 (Vintage 2009 vs Vintage 2020)",
            "hypothesis": "H0: No difference in mean (unequal variance assumed)",
            "test_statistic": t_stat_w1,
            "statistic_name": "t-statistic",
            "df": "approx",  # Welch-Satterthwaite approximation
            "p_value": p_val_w1,
            "significance_level": 0.05,
            "reject_null": p_val_w1 < 0.05,
            "effect_size": cohens_d_1,
            "effect_size_name": "Cohen's d",
            "interpretation": "Significant" if p_val_w1 < 0.05 else "Not significant",
            "notes": "Robust to unequal variances",
        }
    )

    results.append(
        {
            "test_id": "LS-T1-MW",
            "test_name": "Mann-Whitney U test",
            "transition": "2009→2010 (Vintage 2009 vs Vintage 2020)",
            "hypothesis": "H0: Distributions are equal (non-parametric)",
            "test_statistic": u_stat_1,
            "statistic_name": "U-statistic",
            "df": "N/A",
            "p_value": p_val_mw1,
            "significance_level": 0.05,
            "reject_null": p_val_mw1 < 0.05,
            "effect_size": u_stat_1 / (len(v2009) * len(v2020)),  # Common language effect size
            "effect_size_name": "Rank-biserial r",
            "interpretation": "Significant" if p_val_mw1 < 0.05 else "Not significant",
            "notes": "Non-parametric alternative, robust to non-normality",
        }
    )

    # --- Transition 2: 2019 -> 2020 (Vintage 2020 to Vintage 2024) ---
    t_stat_2, p_val_t_2 = ttest_ind(v2020, v2024)
    t_stat_w2, p_val_w2 = ttest_ind(v2020, v2024, equal_var=False)
    u_stat_2, p_val_mw2 = mannwhitneyu(v2020, v2024, alternative="two-sided")
    cohens_d_2 = compute_effect_size_cohens_d(v2020, v2024)

    mean_v2024 = np.mean(v2024)
    diff_2 = mean_v2024 - mean_v2020

    results.append(
        {
            "test_id": "LS-T2-TTEST",
            "test_name": "Two-sample t-test",
            "transition": "2019→2020 (Vintage 2020 vs Vintage 2024)",
            "hypothesis": "H0: No difference in mean international migration between Vintage 2020 and Vintage 2024",
            "test_statistic": t_stat_2,
            "statistic_name": "t-statistic",
            "df": len(v2020) + len(v2024) - 2,
            "p_value": p_val_t_2,
            "significance_level": 0.05,
            "reject_null": p_val_t_2 < 0.05,
            "effect_size": cohens_d_2,
            "effect_size_name": "Cohen's d",
            "interpretation": "Significant" if p_val_t_2 < 0.05 else "Not significant",
            "notes": f"Mean V2020={mean_v2020:.1f}, Mean V2024={mean_v2024:.1f}, Diff={diff_2:.1f}. CAUTION: COVID-19 confound",
        }
    )

    results.append(
        {
            "test_id": "LS-T2-WELCH",
            "test_name": "Welch's t-test",
            "transition": "2019→2020 (Vintage 2020 vs Vintage 2024)",
            "hypothesis": "H0: No difference in mean (unequal variance assumed)",
            "test_statistic": t_stat_w2,
            "statistic_name": "t-statistic",
            "df": "approx",
            "p_value": p_val_w2,
            "significance_level": 0.05,
            "reject_null": p_val_w2 < 0.05,
            "effect_size": cohens_d_2,
            "effect_size_name": "Cohen's d",
            "interpretation": "Significant" if p_val_w2 < 0.05 else "Not significant",
            "notes": "CAUTION: COVID-19 confound makes interpretation difficult",
        }
    )

    results.append(
        {
            "test_id": "LS-T2-MW",
            "test_name": "Mann-Whitney U test",
            "transition": "2019→2020 (Vintage 2020 vs Vintage 2024)",
            "hypothesis": "H0: Distributions are equal (non-parametric)",
            "test_statistic": u_stat_2,
            "statistic_name": "U-statistic",
            "df": "N/A",
            "p_value": p_val_mw2,
            "significance_level": 0.05,
            "reject_null": p_val_mw2 < 0.05,
            "effect_size": u_stat_2 / (len(v2020) * len(v2024)),
            "effect_size_name": "Rank-biserial r",
            "interpretation": "Significant" if p_val_mw2 < 0.05 else "Not significant",
            "notes": "CAUTION: COVID-19 confound",
        }
    )

    # Summary statistics
    summary = {
        "vintage_2009": {
            "years": "2000-2009",
            "n": len(v2009),
            "mean": float(mean_v2009),
            "std": float(np.std(v2009, ddof=1)),
            "median": float(np.median(v2009)),
            "min": float(np.min(v2009)),
            "max": float(np.max(v2009)),
        },
        "vintage_2020": {
            "years": "2010-2019",
            "n": len(v2020),
            "mean": float(mean_v2020),
            "std": float(np.std(v2020, ddof=1)),
            "median": float(np.median(v2020)),
            "min": float(np.min(v2020)),
            "max": float(np.max(v2020)),
        },
        "vintage_2024": {
            "years": "2020-2024",
            "n": len(v2024),
            "mean": float(mean_v2024),
            "std": float(np.std(v2024, ddof=1)),
            "median": float(np.median(v2024)),
            "min": float(np.min(v2024)),
            "max": float(np.max(v2024)),
        },
        "transition_1_diff": float(diff_1),
        "transition_2_diff": float(diff_2),
    }

    return results, summary


def variance_analysis(nd_df):
    """
    Test for heteroskedasticity across vintages using Levene's and Bartlett's tests.
    """
    results = []

    v2009 = nd_df[nd_df["vintage"] == 2009]["INTERNATIONALMIG"].values
    v2020 = nd_df[nd_df["vintage"] == 2020]["INTERNATIONALMIG"].values
    v2024 = nd_df[nd_df["vintage"] == 2024]["INTERNATIONALMIG"].values

    # Calculate variances
    var_2009 = np.var(v2009, ddof=1)
    var_2020 = np.var(v2020, ddof=1)
    var_2024 = np.var(v2024, ddof=1)

    # Levene's test (robust to non-normality)
    levene_stat, levene_p = levene(v2009, v2020, v2024)

    results.append(
        {
            "test_id": "VAR-LEVENE-ALL",
            "test_name": "Levene's test (all vintages)",
            "hypothesis": "H0: Variances are equal across all three vintages",
            "test_statistic": levene_stat,
            "statistic_name": "W-statistic",
            "df": f"({2}, {len(v2009)+len(v2020)+len(v2024)-3})",
            "p_value": levene_p,
            "significance_level": 0.05,
            "reject_null": levene_p < 0.05,
            "effect_size": max(var_2009, var_2020, var_2024) / min(var_2009, var_2020, var_2024)
            if min(var_2009, var_2020, var_2024) > 0
            else np.inf,
            "effect_size_name": "Variance ratio (max/min)",
            "interpretation": "Heteroskedastic" if levene_p < 0.05 else "Homoskedastic",
            "notes": f"Var2009={var_2009:.1f}, Var2020={var_2020:.1f}, Var2024={var_2024:.1f}",
        }
    )

    # Bartlett's test (assumes normality)
    bartlett_stat, bartlett_p = bartlett(v2009, v2020, v2024)

    results.append(
        {
            "test_id": "VAR-BARTLETT-ALL",
            "test_name": "Bartlett's test (all vintages)",
            "hypothesis": "H0: Variances are equal across all three vintages (assumes normality)",
            "test_statistic": bartlett_stat,
            "statistic_name": "Chi-squared statistic",
            "df": 2,
            "p_value": bartlett_p,
            "significance_level": 0.05,
            "reject_null": bartlett_p < 0.05,
            "effect_size": max(var_2009, var_2020, var_2024) / min(var_2009, var_2020, var_2024)
            if min(var_2009, var_2020, var_2024) > 0
            else np.inf,
            "effect_size_name": "Variance ratio (max/min)",
            "interpretation": "Heteroskedastic" if bartlett_p < 0.05 else "Homoskedastic",
            "notes": "Sensitive to non-normality",
        }
    )

    # Pairwise tests: Vintage 2009 vs Vintage 2020
    levene_12, levene_p12 = levene(v2009, v2020)
    results.append(
        {
            "test_id": "VAR-LEVENE-T1",
            "test_name": "Levene's test (Vintage 2009 vs 2020)",
            "hypothesis": "H0: Variances are equal between Vintage 2009 and Vintage 2020",
            "test_statistic": levene_12,
            "statistic_name": "W-statistic",
            "df": f"(1, {len(v2009)+len(v2020)-2})",
            "p_value": levene_p12,
            "significance_level": 0.05,
            "reject_null": levene_p12 < 0.05,
            "effect_size": var_2020 / var_2009 if var_2009 > 0 else np.inf,
            "effect_size_name": "Variance ratio (V2020/V2009)",
            "interpretation": "Heteroskedastic" if levene_p12 < 0.05 else "Homoskedastic",
            "notes": "First transition: 2009→2010",
        }
    )

    # Pairwise tests: Vintage 2020 vs Vintage 2024
    levene_23, levene_p23 = levene(v2020, v2024)
    results.append(
        {
            "test_id": "VAR-LEVENE-T2",
            "test_name": "Levene's test (Vintage 2020 vs 2024)",
            "hypothesis": "H0: Variances are equal between Vintage 2020 and Vintage 2024",
            "test_statistic": levene_23,
            "statistic_name": "W-statistic",
            "df": f"(1, {len(v2020)+len(v2024)-2})",
            "p_value": levene_p23,
            "significance_level": 0.05,
            "reject_null": levene_p23 < 0.05,
            "effect_size": var_2024 / var_2020 if var_2020 > 0 else np.inf,
            "effect_size_name": "Variance ratio (V2024/V2020)",
            "interpretation": "Heteroskedastic" if levene_p23 < 0.05 else "Homoskedastic",
            "notes": "Second transition: 2019→2020 (COVID confound)",
        }
    )

    variance_summary = {
        "vintage_2009": {
            "variance": float(var_2009),
            "std": float(np.sqrt(var_2009)),
            "n": len(v2009),
        },
        "vintage_2020": {
            "variance": float(var_2020),
            "std": float(np.sqrt(var_2020)),
            "n": len(v2020),
        },
        "vintage_2024": {
            "variance": float(var_2024),
            "std": float(np.sqrt(var_2024)),
            "n": len(v2024),
        },
        "variance_ratio_max_min": float(
            max(var_2009, var_2020, var_2024) / min(var_2009, var_2020, var_2024)
        )
        if min(var_2009, var_2020, var_2024) > 0
        else None,
    }

    return results, variance_summary


def chow_test(y, X, break_point):
    """
    Perform Chow test for structural break at a known break point.

    Parameters:
    - y: dependent variable (array)
    - X: independent variable(s) with constant (array)
    - break_point: index where break occurs

    Returns:
    - F-statistic, p-value, RSS values
    """
    n = len(y)
    k = X.shape[1]  # number of parameters

    # Full sample regression
    model_full = OLS(y, X).fit()
    RSS_full = np.sum(model_full.resid**2)

    # Before break
    y1, X1 = y[:break_point], X[:break_point]
    model1 = OLS(y1, X1).fit()
    RSS1 = np.sum(model1.resid**2)

    # After break
    y2, X2 = y[break_point:], X[break_point:]
    model2 = OLS(y2, X2).fit()
    RSS2 = np.sum(model2.resid**2)

    # Chow F-statistic
    RSS_unrestricted = RSS1 + RSS2
    df1 = k  # number of restrictions
    df2 = n - 2 * k  # residual df

    if df2 <= 0:
        return np.nan, np.nan, RSS_full, RSS_unrestricted

    F_stat = ((RSS_full - RSS_unrestricted) / df1) / (RSS_unrestricted / df2)
    p_value = 1 - stats.f.cdf(F_stat, df1, df2)

    return F_stat, p_value, RSS_full, RSS_unrestricted


def structural_break_analysis(nd_df):
    """
    Perform Chow tests at known transition points and placebo tests at other years.
    """
    results = []

    years = nd_df["year"].values
    y = nd_df["INTERNATIONALMIG"].values
    X = add_constant(np.arange(len(y)))  # Simple linear trend

    # Actual transition indices
    # 2009→2010 transition: year 2009 is index 9, year 2010 is index 10
    transition1_idx = 10  # After 2009 (first year of Vintage 2020)
    # 2019→2020 transition: year 2019 is index 19, year 2020 is index 20
    transition2_idx = 20  # After 2019 (first year of Vintage 2024)

    # Chow test at transition 1 (2009→2010)
    F1, p1, RSS_full1, RSS_unr1 = chow_test(y, X, transition1_idx)
    results.append(
        {
            "test_id": "CHOW-T1",
            "test_name": "Chow test at 2009→2010 transition",
            "hypothesis": "H0: No structural break at vintage transition",
            "test_statistic": F1,
            "statistic_name": "F-statistic",
            "df": "(2, 21)",  # k=2, n-2k = 25-4 = 21
            "p_value": p1,
            "significance_level": 0.05,
            "reject_null": p1 < 0.05 if not np.isnan(p1) else False,
            "effect_size": (RSS_full1 - RSS_unr1) / RSS_full1 if RSS_full1 > 0 else np.nan,
            "effect_size_name": "R-squared change",
            "interpretation": "Structural break detected" if p1 < 0.05 else "No structural break",
            "notes": "Clean vintage transition (no COVID)",
        }
    )

    # Chow test at transition 2 (2019→2020)
    F2, p2, RSS_full2, RSS_unr2 = chow_test(y, X, transition2_idx)
    results.append(
        {
            "test_id": "CHOW-T2",
            "test_name": "Chow test at 2019→2020 transition",
            "hypothesis": "H0: No structural break at vintage transition",
            "test_statistic": F2,
            "statistic_name": "F-statistic",
            "df": "(2, 21)",
            "p_value": p2,
            "significance_level": 0.05,
            "reject_null": p2 < 0.05 if not np.isnan(p2) else False,
            "effect_size": (RSS_full2 - RSS_unr2) / RSS_full2 if RSS_full2 > 0 else np.nan,
            "effect_size_name": "R-squared change",
            "interpretation": "Structural break detected" if p2 < 0.05 else "No structural break",
            "notes": "CAUTION: COVID confound",
        }
    )

    # Placebo tests at non-transition years
    placebo_years = [5, 7, 12, 15, 17]  # indices for years 2005, 2007, 2012, 2015, 2017
    placebo_results = []

    for idx in placebo_years:
        if idx >= 3 and idx <= len(y) - 3:  # Ensure enough data on both sides
            F_p, p_p, _, _ = chow_test(y, X, idx)
            year_label = years[idx]
            placebo_results.append(
                {
                    "year": int(year_label),
                    "index": idx,
                    "F_statistic": float(F_p) if not np.isnan(F_p) else None,
                    "p_value": float(p_p) if not np.isnan(p_p) else None,
                }
            )
            results.append(
                {
                    "test_id": f"CHOW-PLACEBO-{year_label}",
                    "test_name": f"Chow test (placebo) at year {year_label}",
                    "hypothesis": "H0: No structural break at non-transition year",
                    "test_statistic": F_p,
                    "statistic_name": "F-statistic",
                    "df": "(2, 21)",
                    "p_value": p_p,
                    "significance_level": 0.05,
                    "reject_null": p_p < 0.05 if not np.isnan(p_p) else False,
                    "effect_size": np.nan,
                    "effect_size_name": "N/A",
                    "interpretation": "Placebo break detected"
                    if p_p < 0.05
                    else "No break (expected)",
                    "notes": "Placebo test - should NOT be significant if vintage transitions are artifacts",
                }
            )

    break_summary = {
        "transition_1": {
            "break_point": "2009→2010",
            "F_statistic": float(F1) if not np.isnan(F1) else None,
            "p_value": float(p1) if not np.isnan(p1) else None,
            "significant": bool(p1 < 0.05) if not np.isnan(p1) else None,
        },
        "transition_2": {
            "break_point": "2019→2020",
            "F_statistic": float(F2) if not np.isnan(F2) else None,
            "p_value": float(p2) if not np.isnan(p2) else None,
            "significant": bool(p2 < 0.05) if not np.isnan(p2) else None,
            "caveat": "COVID-19 confound",
        },
        "placebo_tests": placebo_results,
    }

    return results, break_summary


def autocorrelation_analysis(nd_df):
    """
    Analyze ACF/PACF patterns within each vintage and test for structure changes.
    """
    results = []

    v2009 = nd_df[nd_df["vintage"] == 2009]["INTERNATIONALMIG"].values
    v2020 = nd_df[nd_df["vintage"] == 2020]["INTERNATIONALMIG"].values
    v2024 = nd_df[nd_df["vintage"] == 2024]["INTERNATIONALMIG"].values
    full_series = nd_df["INTERNATIONALMIG"].values

    acf_results = {}

    # Full series ACF (up to 5 lags for n=25)
    nlags_full = min(5, len(full_series) // 4)
    acf_full = acf(full_series, nlags=nlags_full, fft=False)

    acf_results["full_series"] = {
        "n": len(full_series),
        "max_lag": nlags_full,
        "acf_values": [float(x) for x in acf_full],
        "critical_value_95": 1.96 / np.sqrt(len(full_series)),
    }

    # ACF for each vintage (limited by small n)
    for name, data in [("vintage_2009", v2009), ("vintage_2020", v2020), ("vintage_2024", v2024)]:
        if len(data) >= 4:
            nlags = min(3, len(data) // 3)
            try:
                acf_vals = acf(data, nlags=nlags, fft=False)
                acf_results[name] = {
                    "n": len(data),
                    "max_lag": nlags,
                    "acf_values": [float(x) for x in acf_vals],
                    "critical_value_95": 1.96 / np.sqrt(len(data)),
                }
            except Exception:
                acf_results[name] = {"n": len(data), "error": "Insufficient data for ACF"}
        else:
            acf_results[name] = {"n": len(data), "error": "Insufficient data for ACF"}

    # Ljung-Box test for autocorrelation in full series
    try:
        lb_result = acorr_ljungbox(full_series, lags=[1, 2, 3], return_df=True)
        for lag in [1, 2, 3]:
            lb_stat = lb_result.loc[lag, "lb_stat"]
            lb_pval = lb_result.loc[lag, "lb_pvalue"]
            results.append(
                {
                    "test_id": f"ACF-LB-LAG{lag}",
                    "test_name": f"Ljung-Box test (lag {lag})",
                    "hypothesis": f"H0: No autocorrelation up to lag {lag}",
                    "test_statistic": lb_stat,
                    "statistic_name": "Q-statistic",
                    "df": lag,
                    "p_value": lb_pval,
                    "significance_level": 0.05,
                    "reject_null": lb_pval < 0.05,
                    "effect_size": acf_full[lag] if lag < len(acf_full) else np.nan,
                    "effect_size_name": f"ACF at lag {lag}",
                    "interpretation": "Autocorrelation present"
                    if lb_pval < 0.05
                    else "No significant autocorrelation",
                    "notes": "Full series (n=25)",
                }
            )
    except Exception as e:
        results.append(
            {
                "test_id": "ACF-LB-ERROR",
                "test_name": "Ljung-Box test",
                "hypothesis": "N/A",
                "test_statistic": np.nan,
                "statistic_name": "N/A",
                "df": "N/A",
                "p_value": np.nan,
                "significance_level": 0.05,
                "reject_null": False,
                "effect_size": np.nan,
                "effect_size_name": "N/A",
                "interpretation": f"Error: {str(e)}",
                "notes": "Test failed",
            }
        )

    # Test for change in autocorrelation structure across vintages
    # Compare first-order autocorrelation across vintages
    autocorr_by_vintage = {}
    for name, data in [("vintage_2009", v2009), ("vintage_2020", v2020)]:
        if len(data) >= 4:
            autocorr_by_vintage[name] = float(np.corrcoef(data[:-1], data[1:])[0, 1])
        else:
            autocorr_by_vintage[name] = None

    acf_results["first_order_autocorr_by_vintage"] = autocorr_by_vintage

    return results, acf_results


def trend_analysis(nd_df):
    """
    Fit linear trends within each vintage and compare slopes.
    """
    results = []
    trend_results = {}

    v2009_df = nd_df[nd_df["vintage"] == 2009].copy()
    v2020_df = nd_df[nd_df["vintage"] == 2020].copy()
    v2024_df = nd_df[nd_df["vintage"] == 2024].copy()

    for name, df in [
        ("vintage_2009", v2009_df),
        ("vintage_2020", v2020_df),
        ("vintage_2024", v2024_df),
    ]:
        if len(df) >= 3:
            y = df["INTERNATIONALMIG"].values
            X = add_constant(np.arange(len(y)))
            model = OLS(y, X).fit()

            trend_results[name] = {
                "n": len(y),
                "slope": float(model.params[1]),
                "slope_se": float(model.bse[1]),
                "slope_pvalue": float(model.pvalues[1]),
                "intercept": float(model.params[0]),
                "r_squared": float(model.rsquared),
                "significant_trend": bool(model.pvalues[1] < 0.05),
            }

            results.append(
                {
                    "test_id": f"TREND-{name.upper()}",
                    "test_name": f"Linear trend test ({name})",
                    "hypothesis": "H0: Slope = 0 (no linear trend)",
                    "test_statistic": model.tvalues[1],
                    "statistic_name": "t-statistic",
                    "df": len(y) - 2,
                    "p_value": model.pvalues[1],
                    "significance_level": 0.05,
                    "reject_null": model.pvalues[1] < 0.05,
                    "effect_size": model.rsquared,
                    "effect_size_name": "R-squared",
                    "interpretation": f"Slope = {model.params[1]:.1f} persons/year",
                    "notes": "Trend significant"
                    if model.pvalues[1] < 0.05
                    else "No significant trend",
                }
            )

    # Compare slopes between vintages (informal - small sample caveat)
    slope_2009 = trend_results.get("vintage_2009", {}).get("slope", np.nan)
    slope_2020 = trend_results.get("vintage_2020", {}).get("slope", np.nan)
    slope_2024 = trend_results.get("vintage_2024", {}).get("slope", np.nan)

    trend_results["slope_comparison"] = {
        "slope_vintage_2009": slope_2009,
        "slope_vintage_2020": slope_2020,
        "slope_vintage_2024": slope_2024,
        "slope_change_at_t1": slope_2020 - slope_2009
        if not (np.isnan(slope_2009) or np.isnan(slope_2020))
        else None,
        "slope_change_at_t2": slope_2024 - slope_2020
        if not (np.isnan(slope_2020) or np.isnan(slope_2024))
        else None,
        "note": "Small samples limit formal slope difference tests",
    }

    # Full series trend
    y_full = nd_df["INTERNATIONALMIG"].values
    X_full = add_constant(np.arange(len(y_full)))
    model_full = OLS(y_full, X_full).fit()

    trend_results["full_series"] = {
        "n": len(y_full),
        "slope": float(model_full.params[1]),
        "slope_se": float(model_full.bse[1]),
        "slope_pvalue": float(model_full.pvalues[1]),
        "intercept": float(model_full.params[0]),
        "r_squared": float(model_full.rsquared),
    }

    results.append(
        {
            "test_id": "TREND-FULL",
            "test_name": "Linear trend test (full series)",
            "hypothesis": "H0: Slope = 0 (no linear trend)",
            "test_statistic": model_full.tvalues[1],
            "statistic_name": "t-statistic",
            "df": len(y_full) - 2,
            "p_value": model_full.pvalues[1],
            "significance_level": 0.05,
            "reject_null": model_full.pvalues[1] < 0.05,
            "effect_size": model_full.rsquared,
            "effect_size_name": "R-squared",
            "interpretation": f"Slope = {model_full.params[1]:.1f} persons/year across full 25-year series",
            "notes": "Overall trend ignoring vintage boundaries",
        }
    )

    return results, trend_results


def create_visualizations(nd_df, variance_summary, break_summary, acf_results):
    """Create all required visualizations."""

    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: Time series with vintage boundaries
    fig1, ax1 = plt.subplots(figsize=(12, 6), dpi=300)

    colors = {"2009": "#1f77b4", "2020": "#ff7f0e", "2024": "#2ca02c"}

    for vintage, color in colors.items():
        mask = nd_df["vintage"] == int(vintage)
        subset = nd_df[mask]
        ax1.plot(
            subset["year"],
            subset["INTERNATIONALMIG"],
            "o-",
            color=color,
            linewidth=2,
            markersize=8,
            label=f"Vintage {vintage}",
        )

    # Add vertical lines at transitions
    ax1.axvline(
        x=2009.5, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Vintage transition"
    )
    ax1.axvline(x=2019.5, color="red", linestyle="--", linewidth=2, alpha=0.7)

    # Add annotations
    ax1.annotate(
        "Vintage 2009\n(2000-2009)",
        xy=(2004, ax1.get_ylim()[1] * 0.9),
        fontsize=10,
        ha="center",
        color="#1f77b4",
    )
    ax1.annotate(
        "Vintage 2020\n(2010-2019)",
        xy=(2014, ax1.get_ylim()[1] * 0.9),
        fontsize=10,
        ha="center",
        color="#ff7f0e",
    )
    ax1.annotate(
        "Vintage 2024\n(2020-2024)",
        xy=(2022, ax1.get_ylim()[1] * 0.9),
        fontsize=10,
        ha="center",
        color="#2ca02c",
    )

    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("International Migration (persons)", fontsize=12)
    ax1.set_title(
        "North Dakota Net International Migration by Census Vintage", fontsize=14, fontweight="bold"
    )
    ax1.legend(loc="upper left")
    ax1.set_xlim(1999, 2025)

    plt.tight_layout()
    fig1.savefig(
        OUTPUT_DIR / "agent2_fig1_timeseries_with_vintages.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig1)

    # Figure 2: Variance comparison
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Box plot
    vintage_data = [
        nd_df[nd_df["vintage"] == 2009]["INTERNATIONALMIG"].values,
        nd_df[nd_df["vintage"] == 2020]["INTERNATIONALMIG"].values,
        nd_df[nd_df["vintage"] == 2024]["INTERNATIONALMIG"].values,
    ]

    bp = ax2a.boxplot(
        vintage_data,
        labels=["Vintage 2009\n(n=10)", "Vintage 2020\n(n=10)", "Vintage 2024\n(n=5)"],
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], ["#1f77b4", "#ff7f0e", "#2ca02c"], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax2a.set_ylabel("International Migration (persons)", fontsize=12)
    ax2a.set_title("Distribution by Vintage", fontsize=12, fontweight="bold")

    # Variance bar chart
    variances = [
        variance_summary["vintage_2009"]["variance"],
        variance_summary["vintage_2020"]["variance"],
        variance_summary["vintage_2024"]["variance"],
    ]
    stds = [np.sqrt(v) for v in variances]

    bars = ax2b.bar(
        ["Vintage 2009", "Vintage 2020", "Vintage 2024"],
        variances,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        alpha=0.7,
    )
    ax2b.set_ylabel("Variance", fontsize=12)
    ax2b.set_title("Variance Comparison Across Vintages", fontsize=12, fontweight="bold")

    # Add variance values on bars
    for bar, var in zip(bars, variances, strict=False):
        height = bar.get_height()
        ax2b.annotate(
            f"{var:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    fig2.savefig(OUTPUT_DIR / "agent2_fig2_variance_by_vintage.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: Structural break analysis
    fig3, ax3 = plt.subplots(figsize=(12, 6), dpi=300)

    years = nd_df["year"].values
    y = nd_df["INTERNATIONALMIG"].values

    ax3.plot(years, y, "ko-", linewidth=1.5, markersize=6, label="Observed")

    # Highlight transition points
    ax3.axvline(x=2009.5, color="red", linestyle="--", linewidth=2, alpha=0.8)
    ax3.axvline(x=2019.5, color="red", linestyle="--", linewidth=2, alpha=0.8)

    # Add Chow test results as text
    t1_result = break_summary["transition_1"]
    t2_result = break_summary["transition_2"]

    t1_text = (
        f"2009→2010 Chow test:\nF={t1_result['F_statistic']:.2f}, p={t1_result['p_value']:.3f}"
    )
    t2_text = f"2019→2020 Chow test:\nF={t2_result['F_statistic']:.2f}, p={t2_result['p_value']:.3f}\n(COVID confound)"

    ax3.annotate(
        t1_text,
        xy=(2009.5, max(y) * 0.7),
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        ha="center",
    )
    ax3.annotate(
        t2_text,
        xy=(2019.5, max(y) * 0.5),
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        ha="center",
    )

    ax3.set_xlabel("Year", fontsize=12)
    ax3.set_ylabel("International Migration (persons)", fontsize=12)
    ax3.set_title(
        "Structural Break Analysis at Vintage Transitions", fontsize=14, fontweight="bold"
    )
    ax3.legend()

    plt.tight_layout()
    fig3.savefig(OUTPUT_DIR / "agent2_fig3_structural_breaks.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # Figure 4: ACF by vintage
    fig4, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

    # Full series ACF
    full_acf = acf_results.get("full_series", {})
    if "acf_values" in full_acf:
        acf_vals = full_acf["acf_values"]
        lags = range(len(acf_vals))
        crit = full_acf["critical_value_95"]

        axes[0, 0].bar(lags, acf_vals, color="steelblue", alpha=0.7)
        axes[0, 0].axhline(y=crit, color="red", linestyle="--", label=f"95% CI ({crit:.3f})")
        axes[0, 0].axhline(y=-crit, color="red", linestyle="--")
        axes[0, 0].axhline(y=0, color="black", linewidth=0.5)
        axes[0, 0].set_title("Full Series ACF (n=25)", fontweight="bold")
        axes[0, 0].set_xlabel("Lag")
        axes[0, 0].set_ylabel("ACF")
        axes[0, 0].legend()

    # Vintage 2009 ACF
    v2009_acf = acf_results.get("vintage_2009", {})
    if "acf_values" in v2009_acf:
        acf_vals = v2009_acf["acf_values"]
        lags = range(len(acf_vals))
        crit = v2009_acf["critical_value_95"]

        axes[0, 1].bar(lags, acf_vals, color="#1f77b4", alpha=0.7)
        axes[0, 1].axhline(y=crit, color="red", linestyle="--", label=f"95% CI ({crit:.3f})")
        axes[0, 1].axhline(y=-crit, color="red", linestyle="--")
        axes[0, 1].axhline(y=0, color="black", linewidth=0.5)
        axes[0, 1].set_title("Vintage 2009 ACF (n=10)", fontweight="bold")
        axes[0, 1].set_xlabel("Lag")
        axes[0, 1].set_ylabel("ACF")
        axes[0, 1].legend()
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "Insufficient data for ACF",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].set_title("Vintage 2009 ACF (n=10)", fontweight="bold")

    # Vintage 2020 ACF
    v2020_acf = acf_results.get("vintage_2020", {})
    if "acf_values" in v2020_acf:
        acf_vals = v2020_acf["acf_values"]
        lags = range(len(acf_vals))
        crit = v2020_acf["critical_value_95"]

        axes[1, 0].bar(lags, acf_vals, color="#ff7f0e", alpha=0.7)
        axes[1, 0].axhline(y=crit, color="red", linestyle="--", label=f"95% CI ({crit:.3f})")
        axes[1, 0].axhline(y=-crit, color="red", linestyle="--")
        axes[1, 0].axhline(y=0, color="black", linewidth=0.5)
        axes[1, 0].set_title("Vintage 2020 ACF (n=10)", fontweight="bold")
        axes[1, 0].set_xlabel("Lag")
        axes[1, 0].set_ylabel("ACF")
        axes[1, 0].legend()
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "Insufficient data for ACF",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_title("Vintage 2020 ACF (n=10)", fontweight="bold")

    # Vintage 2024 ACF
    v2024_acf = acf_results.get("vintage_2024", {})
    if "acf_values" in v2024_acf:
        acf_vals = v2024_acf["acf_values"]
        lags = range(len(acf_vals))
        crit = v2024_acf["critical_value_95"]

        axes[1, 1].bar(lags, acf_vals, color="#2ca02c", alpha=0.7)
        axes[1, 1].axhline(y=crit, color="red", linestyle="--", label=f"95% CI ({crit:.3f})")
        axes[1, 1].axhline(y=-crit, color="red", linestyle="--")
        axes[1, 1].axhline(y=0, color="black", linewidth=0.5)
        axes[1, 1].set_title("Vintage 2024 ACF (n=5)", fontweight="bold")
        axes[1, 1].set_xlabel("Lag")
        axes[1, 1].set_ylabel("ACF")
        axes[1, 1].legend()
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Insufficient data for ACF\n(n=5 too small)",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_title("Vintage 2024 ACF (n=5)", fontweight="bold")

    plt.suptitle(
        "Autocorrelation Function (ACF) Analysis by Vintage", fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig4.savefig(OUTPUT_DIR / "agent2_fig4_acf_by_vintage.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)

    print("All visualizations saved.")


def generate_outputs(
    nd_df, all_results, level_summary, variance_summary, break_summary, acf_results, trend_results
):
    """Generate all required output files."""

    # 1. Export ND data
    nd_export = nd_df[["year", "INTERNATIONALMIG", "vintage", "vintage_period"]].copy()
    nd_export.columns = ["year", "intl_migration", "vintage", "vintage_period"]
    nd_export.to_csv(OUTPUT_DIR / "agent2_nd_migration_data.csv", index=False)
    print("Saved: agent2_nd_migration_data.csv")

    # 2. Export test results
    test_df = pd.DataFrame(all_results)
    # Ensure proper column order
    cols = [
        "test_id",
        "test_name",
        "hypothesis",
        "test_statistic",
        "statistic_name",
        "df",
        "p_value",
        "significance_level",
        "reject_null",
        "effect_size",
        "effect_size_name",
        "interpretation",
        "notes",
    ]
    # Add missing columns if any
    for col in cols:
        if col not in test_df.columns:
            test_df[col] = ""
    test_df = test_df[cols]
    test_df.to_csv(OUTPUT_DIR / "agent2_test_results.csv", index=False)
    print("Saved: agent2_test_results.csv")

    # 3. Transition metrics JSON
    transition_metrics = {
        "generated_date": datetime.now().isoformat(),
        "agent": 2,
        "level_shift_analysis": level_summary,
        "variance_analysis": variance_summary,
        "structural_break_analysis": break_summary,
        "autocorrelation_analysis": acf_results,
        "trend_analysis": trend_results,
    }

    with open(OUTPUT_DIR / "agent2_transition_metrics.json", "w") as f:
        json.dump(transition_metrics, f, indent=2, default=str)
    print("Saved: agent2_transition_metrics.json")

    # 4. Findings summary JSON
    findings = {
        "agent_id": 2,
        "report_date": datetime.now().strftime("%Y-%m-%d"),
        "primary_question": "Is there statistical evidence of vintage-related artifacts in the ND international migration time series?",
        "key_findings": {
            "level_shift_2009_2010": {
                "mean_difference": level_summary["transition_1_diff"],
                "ttest_pvalue": next(
                    (r["p_value"] for r in all_results if r["test_id"] == "LS-T1-TTEST"), None
                ),
                "welch_pvalue": next(
                    (r["p_value"] for r in all_results if r["test_id"] == "LS-T1-WELCH"), None
                ),
                "significant": next(
                    (r["reject_null"] for r in all_results if r["test_id"] == "LS-T1-TTEST"), None
                ),
                "interpretation": "Significant mean increase at first vintage transition"
                if next(
                    (r["reject_null"] for r in all_results if r["test_id"] == "LS-T1-TTEST"), False
                )
                else "No significant level shift detected",
            },
            "level_shift_2019_2020": {
                "mean_difference": level_summary["transition_2_diff"],
                "ttest_pvalue": next(
                    (r["p_value"] for r in all_results if r["test_id"] == "LS-T2-TTEST"), None
                ),
                "significant": next(
                    (r["reject_null"] for r in all_results if r["test_id"] == "LS-T2-TTEST"), None
                ),
                "interpretation": "Confounded by COVID-19 pandemic",
                "caveat": "Cannot separate vintage effect from COVID effect",
            },
            "variance_heterogeneity": {
                "levene_pvalue": next(
                    (r["p_value"] for r in all_results if r["test_id"] == "VAR-LEVENE-ALL"), None
                ),
                "variance_ratio": variance_summary.get("variance_ratio_max_min"),
                "significant": next(
                    (r["reject_null"] for r in all_results if r["test_id"] == "VAR-LEVENE-ALL"),
                    None,
                ),
                "interpretation": "Significant variance differences across vintages"
                if next(
                    (r["reject_null"] for r in all_results if r["test_id"] == "VAR-LEVENE-ALL"),
                    False,
                )
                else "Variances are homogeneous across vintages",
            },
            "structural_break_2009_2010": {
                "chow_F": break_summary["transition_1"]["F_statistic"],
                "chow_pvalue": break_summary["transition_1"]["p_value"],
                "significant": break_summary["transition_1"]["significant"],
                "interpretation": "Structural break detected at first vintage transition"
                if break_summary["transition_1"]["significant"]
                else "No structural break detected",
            },
            "placebo_comparison": {
                "transition_vs_placebo": "See detailed results",
                "note": "Compare F-statistics at transitions vs non-transition years",
            },
        },
        "power_limitations": {
            "sample_size_per_vintage": [10, 10, 5],
            "total_n": 25,
            "note": "Very small samples severely limit statistical power. Failure to reject null does not imply no effect.",
        },
        "overall_assessment": {
            "evidence_of_vintage_artifact": "Inconclusive",
            "confidence": "Low due to small samples",
            "recommendation": "Proceed with caution; consider sensitivity analyses",
            "key_uncertainties": [
                "Small sample sizes (n=10, 10, 5 per vintage)",
                "COVID-19 confounds 2019-2020 transition",
                "Cannot distinguish methodology change from real demographic shifts",
                "Power too low to detect moderate effects",
            ],
        },
    }

    with open(OUTPUT_DIR / "agent2_findings_summary.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)
    print("Saved: agent2_findings_summary.json")

    # 5. Sources JSON
    sources = {
        "statistical_methods": [
            {
                "method": "Two-sample t-test",
                "reference": "Student (1908). The probable error of a mean. Biometrika, 6(1), 1-25.",
                "usage": "Testing mean differences at vintage transitions",
            },
            {
                "method": "Welch's t-test",
                "reference": 'Welch, B.L. (1947). The generalization of "Student\'s" problem. Biometrika, 34(1-2), 28-35.',
                "usage": "Robust mean comparison when variances may differ",
            },
            {
                "method": "Mann-Whitney U test",
                "reference": "Mann, H.B., Whitney, D.R. (1947). On a test of whether one of two random variables is stochastically larger than the other. Annals of Mathematical Statistics, 18(1), 50-60.",
                "usage": "Non-parametric alternative for comparing distributions",
            },
            {
                "method": "Levene's test",
                "reference": "Levene, H. (1960). Robust tests for equality of variances. In Contributions to Probability and Statistics.",
                "usage": "Testing homogeneity of variances across vintages",
            },
            {
                "method": "Bartlett's test",
                "reference": "Bartlett, M.S. (1937). Properties of sufficiency and statistical tests. Proceedings of the Royal Society A, 160(901), 268-282.",
                "usage": "Alternative variance homogeneity test (assumes normality)",
            },
            {
                "method": "Chow test",
                "reference": "Chow, G.C. (1960). Tests of equality between sets of coefficients in two linear regressions. Econometrica, 28(3), 591-605.",
                "usage": "Testing for structural breaks in regression models",
            },
            {
                "method": "Ljung-Box test",
                "reference": "Ljung, G.M., Box, G.E.P. (1978). On a measure of lack of fit in time series models. Biometrika, 65(2), 297-303.",
                "usage": "Testing for autocorrelation in time series",
            },
            {
                "method": "Cohen's d",
                "reference": "Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence Erlbaum Associates.",
                "usage": "Effect size measure for mean differences",
            },
        ],
        "software": [
            {"name": "Python", "version": "3.x", "usage": "Analysis implementation"},
            {"name": "pandas", "usage": "Data manipulation"},
            {"name": "numpy", "usage": "Numerical computations"},
            {"name": "scipy.stats", "usage": "Statistical tests"},
            {"name": "statsmodels", "usage": "Regression and time series analysis"},
            {"name": "matplotlib", "usage": "Visualization"},
        ],
        "data_source": {
            "file": "state_migration_components_2000_2024.csv",
            "path": str(DATA_PATH),
            "description": "Census Bureau population estimates migration components by state and vintage",
        },
    }

    with open(OUTPUT_DIR / "agent2_sources.json", "w") as f:
        json.dump(sources, f, indent=2)
    print("Saved: agent2_sources.json")

    return findings


def generate_calculations_doc(
    nd_df, all_results, level_summary, variance_summary, break_summary, trend_results
):
    """Generate detailed calculations document for verification."""

    v2009 = nd_df[nd_df["vintage"] == 2009]["INTERNATIONALMIG"].values
    v2020 = nd_df[nd_df["vintage"] == 2020]["INTERNATIONALMIG"].values
    v2024 = nd_df[nd_df["vintage"] == 2024]["INTERNATIONALMIG"].values

    doc = f"""# Agent 2: Step-by-Step Calculations

## Data Summary

### Raw Data by Vintage

**Vintage 2009 (2000-2009)**:
Years: {list(nd_df[nd_df['vintage'] == 2009]['year'].values)}
Values: {list(v2009)}

**Vintage 2020 (2010-2019)**:
Years: {list(nd_df[nd_df['vintage'] == 2020]['year'].values)}
Values: {list(v2020)}

**Vintage 2024 (2020-2024)**:
Years: {list(nd_df[nd_df['vintage'] == 2024]['year'].values)}
Values: {list(v2024)}

---

## 1. Descriptive Statistics

### Vintage 2009
- n = {len(v2009)}
- Mean = {np.mean(v2009):.2f}
- Standard Deviation = {np.std(v2009, ddof=1):.2f}
- Variance = {np.var(v2009, ddof=1):.2f}
- Median = {np.median(v2009):.2f}
- Min = {np.min(v2009)}, Max = {np.max(v2009)}

### Vintage 2020
- n = {len(v2020)}
- Mean = {np.mean(v2020):.2f}
- Standard Deviation = {np.std(v2020, ddof=1):.2f}
- Variance = {np.var(v2020, ddof=1):.2f}
- Median = {np.median(v2020):.2f}
- Min = {np.min(v2020)}, Max = {np.max(v2020)}

### Vintage 2024
- n = {len(v2024)}
- Mean = {np.mean(v2024):.2f}
- Standard Deviation = {np.std(v2024, ddof=1):.2f}
- Variance = {np.var(v2024, ddof=1):.2f}
- Median = {np.median(v2024):.2f}
- Min = {np.min(v2024)}, Max = {np.max(v2024)}

---

## 2. Level Shift Analysis (Transition 1: 2009 to 2010)

### Two-Sample t-test

**Hypotheses**:
- H0: mu_V2009 = mu_V2020 (no difference in population means)
- H1: mu_V2009 != mu_V2020 (means differ)

**Calculations**:
- Mean V2009 (x̄1) = {np.mean(v2009):.2f}
- Mean V2020 (x̄2) = {np.mean(v2020):.2f}
- Difference = {np.mean(v2020) - np.mean(v2009):.2f}

- Pooled variance = ((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2)
- s1² = {np.var(v2009, ddof=1):.2f}
- s2² = {np.var(v2020, ddof=1):.2f}
- Pooled variance = (9*{np.var(v2009, ddof=1):.2f} + 9*{np.var(v2020, ddof=1):.2f}) / 18 = {((9*np.var(v2009, ddof=1) + 9*np.var(v2020, ddof=1))/18):.2f}
- Pooled SE = sqrt(pooled_var * (1/n1 + 1/n2)) = {np.sqrt(((9*np.var(v2009, ddof=1) + 9*np.var(v2020, ddof=1))/18) * 0.2):.2f}

- t-statistic = (x̄2 - x̄1) / SE
- df = n1 + n2 - 2 = 18

**Result from scipy.stats.ttest_ind**:
{next((f"t = {r['test_statistic']:.4f}, p = {r['p_value']:.4f}" for r in all_results if r['test_id'] == 'LS-T1-TTEST'), 'N/A')}

### Cohen's d Effect Size

d = (x̄2 - x̄1) / pooled_std
d = {compute_effect_size_cohens_d(v2009, v2020):.3f}

Interpretation:
- |d| < 0.2: negligible
- 0.2 <= |d| < 0.5: small
- 0.5 <= |d| < 0.8: medium
- |d| >= 0.8: large

---

## 3. Variance Analysis

### Within-Vintage Variances
- Var(V2009) = {np.var(v2009, ddof=1):.2f}
- Var(V2020) = {np.var(v2020, ddof=1):.2f}
- Var(V2024) = {np.var(v2024, ddof=1):.2f}

### Levene's Test
Tests equality of variances (robust to non-normality)

H0: sigma1² = sigma2² = sigma3²
H1: At least one variance differs

**Result**:
{next((f"W = {r['test_statistic']:.4f}, p = {r['p_value']:.4f}" for r in all_results if r['test_id'] == 'VAR-LEVENE-ALL'), 'N/A')}

### Variance Ratio
Max/Min ratio = {variance_summary.get('variance_ratio_max_min', 'N/A'):.2f}

---

## 4. Structural Break Analysis (Chow Test)

### Model
y_t = alpha + beta*t + epsilon_t

### Chow Test at 2009/2010 Transition

Break point: t = 10 (year 2010)

**Procedure**:
1. Fit full model (n=25): RSS_full
2. Fit model 1 (2000-2009, n=10): RSS_1
3. Fit model 2 (2010-2024, n=15): RSS_2
4. F = [(RSS_full - (RSS_1 + RSS_2)) / k] / [(RSS_1 + RSS_2) / (n - 2k)]

where k = 2 (parameters: intercept and slope)

**Result**:
F = {break_summary['transition_1']['F_statistic']:.4f}
p = {break_summary['transition_1']['p_value']:.4f}
Significant at alpha=0.05: {break_summary['transition_1']['significant']}

---

## 5. Trend Analysis

### Vintage 2009 Linear Trend
Model: IntlMig = a + b*time

{f"Slope = {trend_results['vintage_2009']['slope']:.2f} persons/year" if 'vintage_2009' in trend_results else 'N/A'}
{f"SE(slope) = {trend_results['vintage_2009']['slope_se']:.2f}" if 'vintage_2009' in trend_results else ''}
{f"p-value = {trend_results['vintage_2009']['slope_pvalue']:.4f}" if 'vintage_2009' in trend_results else ''}
{f"R² = {trend_results['vintage_2009']['r_squared']:.4f}" if 'vintage_2009' in trend_results else ''}

### Vintage 2020 Linear Trend
{f"Slope = {trend_results['vintage_2020']['slope']:.2f} persons/year" if 'vintage_2020' in trend_results else 'N/A'}
{f"SE(slope) = {trend_results['vintage_2020']['slope_se']:.2f}" if 'vintage_2020' in trend_results else ''}
{f"p-value = {trend_results['vintage_2020']['slope_pvalue']:.4f}" if 'vintage_2020' in trend_results else ''}
{f"R² = {trend_results['vintage_2020']['r_squared']:.4f}" if 'vintage_2020' in trend_results else ''}

### Vintage 2024 Linear Trend
{f"Slope = {trend_results['vintage_2024']['slope']:.2f} persons/year" if 'vintage_2024' in trend_results else 'N/A'}
{f"SE(slope) = {trend_results['vintage_2024']['slope_se']:.2f}" if 'vintage_2024' in trend_results else ''}
{f"p-value = {trend_results['vintage_2024']['slope_pvalue']:.4f}" if 'vintage_2024' in trend_results else ''}
{f"R² = {trend_results['vintage_2024']['r_squared']:.4f}" if 'vintage_2024' in trend_results else ''}

---

## 6. Power Analysis Considerations

With n=10 per vintage, statistical power is severely limited:

For a two-sample t-test with n1=n2=10, alpha=0.05, two-tailed:
- To detect large effect (d=0.8): Power ≈ 0.39
- To detect medium effect (d=0.5): Power ≈ 0.18
- To detect small effect (d=0.2): Power ≈ 0.07

**Implication**: Non-significant results should NOT be interpreted as evidence of no effect. The tests have low power to detect even moderate effects.

---

## 7. Key Caveats

1. **Small Sample Sizes**: n=10, 10, and 5 per vintage severely limit statistical power and the validity of asymptotic test assumptions.

2. **COVID-19 Confound**: The 2019-2020 transition coincides with the COVID-19 pandemic, making it impossible to separate vintage methodology effects from genuine pandemic impacts on international migration.

3. **Multiple Testing**: Multiple tests increase Type I error rate. No correction applied; results should be interpreted holistically.

4. **Non-Independence**: Time series data may violate independence assumptions of t-tests. ACF analysis addresses this partially.

5. **Normality Assumption**: Small samples make it difficult to assess normality. Non-parametric alternatives (Mann-Whitney) provided as robustness checks.

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Agent: 2 (Statistical Transition Analysis)*
"""

    with open(OUTPUT_DIR / "agent2_calculations.md", "w") as f:
        f.write(doc)
    print("Saved: agent2_calculations.md")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Agent 2: Statistical Transition Analysis for ADR-019")
    print("=" * 60)

    # Load data
    print("\n1. Loading North Dakota data...")
    nd_df = load_nd_data()
    print(f"   Loaded {len(nd_df)} rows for North Dakota (STATE=38)")
    print(f"   Years: {nd_df['year'].min()} to {nd_df['year'].max()}")
    print(f"   Vintages: {sorted(nd_df['vintage'].unique())}")

    # Initialize results container
    all_results = []

    # Level shift analysis
    print("\n2. Conducting level shift analysis...")
    level_results, level_summary = level_shift_analysis(nd_df)
    all_results.extend(level_results)
    print(f"   Transition 1 (2009→2010) mean diff: {level_summary['transition_1_diff']:.1f}")
    print(f"   Transition 2 (2019→2020) mean diff: {level_summary['transition_2_diff']:.1f}")

    # Variance analysis
    print("\n3. Conducting variance analysis...")
    var_results, variance_summary = variance_analysis(nd_df)
    all_results.extend(var_results)
    print(f"   Variance ratio (max/min): {variance_summary['variance_ratio_max_min']:.2f}")

    # Structural break analysis
    print("\n4. Conducting structural break analysis...")
    break_results, break_summary = structural_break_analysis(nd_df)
    all_results.extend(break_results)
    print(
        f"   Chow test at 2009→2010: F={break_summary['transition_1']['F_statistic']:.2f}, p={break_summary['transition_1']['p_value']:.4f}"
    )
    print(
        f"   Chow test at 2019→2020: F={break_summary['transition_2']['F_statistic']:.2f}, p={break_summary['transition_2']['p_value']:.4f}"
    )

    # Autocorrelation analysis
    print("\n5. Conducting autocorrelation analysis...")
    acf_results_list, acf_results = autocorrelation_analysis(nd_df)
    all_results.extend(acf_results_list)

    # Trend analysis
    print("\n6. Conducting trend analysis...")
    trend_results_list, trend_results = trend_analysis(nd_df)
    all_results.extend(trend_results_list)

    # Create visualizations
    print("\n7. Creating visualizations...")
    create_visualizations(nd_df, variance_summary, break_summary, acf_results)

    # Generate output files
    print("\n8. Generating output files...")
    findings = generate_outputs(
        nd_df,
        all_results,
        level_summary,
        variance_summary,
        break_summary,
        acf_results,
        trend_results,
    )

    # Generate calculations document
    print("\n9. Generating calculations document...")
    generate_calculations_doc(
        nd_df, all_results, level_summary, variance_summary, break_summary, trend_results
    )

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

    # Print summary of key findings
    print("\nKEY FINDINGS SUMMARY:")
    print("-" * 40)
    t1_sig = next((r["reject_null"] for r in all_results if r["test_id"] == "LS-T1-TTEST"), None)
    t2_sig = next((r["reject_null"] for r in all_results if r["test_id"] == "LS-T2-TTEST"), None)
    var_sig = next(
        (r["reject_null"] for r in all_results if r["test_id"] == "VAR-LEVENE-ALL"), None
    )

    print(f"Level shift at 2009→2010: {'SIGNIFICANT' if t1_sig else 'Not significant'}")
    print(
        f"Level shift at 2019→2020: {'SIGNIFICANT' if t2_sig else 'Not significant'} (COVID confound)"
    )
    print(f"Variance heterogeneity: {'SIGNIFICANT' if var_sig else 'Not significant'}")
    print(
        f"Structural break at 2009→2010: {'SIGNIFICANT' if break_summary['transition_1']['significant'] else 'Not significant'}"
    )

    return nd_df, all_results, findings


if __name__ == "__main__":
    main()
