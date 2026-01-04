#!/usr/bin/env python3
"""
Module 3.2: Origin-Country Panel Analysis (Network Effects)

Analyzes diaspora network effects - whether existing immigrant stocks predict new arrivals.

This module examines:
1. Network elasticity estimation (log-log models)
2. Origin-specific growth analysis
3. Diaspora concentration analysis
4. Cross-sectional stock-flow correlations

Author: Statistical Analysis Pipeline
Date: 2025-12-28
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

# Project root relative to this script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[3]  # sdc_2024_replication -> cohort_projections

DATA_DIR = _PROJECT_ROOT / "data/processed/immigration/analysis"
OUTPUT_DIR = _SCRIPT_DIR / "results"
FIGURES_DIR = _SCRIPT_DIR / "figures"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
MIN_POPULATION_THRESHOLD = 50  # Exclude origins with pop < 50 due to noise
LOG_CONSTANT = 1  # Add to avoid log(0)


def load_data():
    """Load all required data files."""
    print("=" * 70)
    print("MODULE 3.2: ORIGIN-COUNTRY PANEL ANALYSIS (NETWORK EFFECTS)")
    print("=" * 70)
    print(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    print("\n[1] Loading Data Files...")

    # Load ACS foreign-born by state and origin
    acs_state_origin = pd.read_parquet(
        DATA_DIR / "acs_foreign_born_by_state_origin.parquet"
    )
    print(f"    - ACS foreign-born by state/origin: {len(acs_state_origin):,} rows")

    # Load ND share data
    acs_nd_share = pd.read_parquet(DATA_DIR / "acs_foreign_born_nd_share.parquet")
    print(f"    - ACS ND share data: {len(acs_nd_share):,} rows")

    # Load DHS LPR by state/country
    dhs_lpr = pd.read_parquet(DATA_DIR / "dhs_lpr_by_state_country.parquet")
    print(f"    - DHS LPR by state/country: {len(dhs_lpr):,} rows")

    return acs_state_origin, acs_nd_share, dhs_lpr


def prepare_panel_data(acs_df):
    """
    Prepare panel data for network effect analysis.
    Filter to country level and create lagged variables.
    """
    print("\n[2] Preparing Panel Data...")

    # Filter to country level only (avoid double-counting with regions)
    panel = acs_df[acs_df["level"] == "country"].copy()
    print(f"    - Country-level observations: {len(panel):,}")

    # Focus on North Dakota
    nd_panel = panel[panel["state_name"] == "North Dakota"].copy()
    print(f"    - North Dakota observations: {len(nd_panel):,}")

    # Sort by country and year for proper lagging
    nd_panel = nd_panel.sort_values(["country", "year"])

    # Create lagged population variable
    nd_panel["foreign_born_pop_lag1"] = nd_panel.groupby("country")[
        "foreign_born_pop"
    ].shift(1)

    # Filter to observations with valid lagged values
    nd_panel_valid = nd_panel[nd_panel["foreign_born_pop_lag1"].notna()].copy()
    print(f"    - Observations with valid lags: {len(nd_panel_valid):,}")

    # Filter out very small populations (noise)
    nd_panel_valid = nd_panel_valid[
        nd_panel_valid["foreign_born_pop_lag1"] >= MIN_POPULATION_THRESHOLD
    ]
    print(
        f"    - After filtering pop >= {MIN_POPULATION_THRESHOLD}: {len(nd_panel_valid):,}"
    )

    # Create log-transformed variables (add constant to handle zeros)
    nd_panel_valid["log_stock"] = np.log(
        nd_panel_valid["foreign_born_pop"] + LOG_CONSTANT
    )
    nd_panel_valid["log_stock_lag1"] = np.log(
        nd_panel_valid["foreign_born_pop_lag1"] + LOG_CONSTANT
    )

    decisions = {
        "filter_level": "country (excluded regions and sub_regions to avoid double-counting)",
        "lag_period": 1,
        "min_population_threshold": MIN_POPULATION_THRESHOLD,
        "log_constant": LOG_CONSTANT,
        "observations_final": len(nd_panel_valid),
        "unique_countries": nd_panel_valid["country"].nunique(),
        "year_range": [
            int(nd_panel_valid["year"].min()),
            int(nd_panel_valid["year"].max()),
        ],
    }

    return nd_panel_valid, panel, decisions


def estimate_network_elasticity(nd_panel):
    """
    Estimate network elasticity using OLS regression.
    Model: log(Stock_t) = alpha + beta * log(Stock_t-1) + epsilon

    The coefficient beta represents the network elasticity:
    - beta = 1: proportional growth (constant percentage increase)
    - beta > 1: accelerating growth (larger communities grow faster)
    - beta < 1: decelerating growth (smaller communities grow faster)
    """
    print("\n[3] Estimating Network Elasticity...")
    print("-" * 70)

    # Prepare data for regression
    X = nd_panel["log_stock_lag1"].values
    y = nd_panel["log_stock"].values
    n = len(X)

    # Add constant for intercept
    X_with_const = np.column_stack([np.ones(n), X])

    # OLS estimation: (X'X)^-1 X'y
    XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
    beta_hat = XtX_inv @ X_with_const.T @ y

    # Predicted values and residuals
    y_hat = X_with_const @ beta_hat
    residuals = y - y_hat

    # Degrees of freedom
    df_model = 1  # one predictor (excluding constant)
    df_residual = n - 2  # n - k - 1 where k = 1 predictor
    df_total = n - 1

    # Sum of squares
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum(residuals**2)
    ss_model = ss_total - ss_residual

    # R-squared and adjusted R-squared
    r_squared = 1 - (ss_residual / ss_total)
    r_squared_adj = 1 - ((1 - r_squared) * (n - 1) / (n - 2))

    # Mean squared error
    mse = ss_residual / df_residual
    rmse = np.sqrt(mse)

    # Standard errors of coefficients
    var_beta = mse * np.diag(XtX_inv)
    se_beta = np.sqrt(var_beta)

    # t-statistics and p-values
    t_stats = beta_hat / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_residual))

    # 95% confidence intervals
    t_critical = stats.t.ppf(0.975, df_residual)
    ci_lower = beta_hat - t_critical * se_beta
    ci_upper = beta_hat + t_critical * se_beta

    # F-statistic for overall model significance
    f_stat = (ss_model / df_model) / (ss_residual / df_residual)
    f_pvalue = 1 - stats.f.cdf(f_stat, df_model, df_residual)

    # Correlation coefficient
    correlation = np.corrcoef(X, y)[0, 1]

    # Print SPSS-style output
    print("\n" + "=" * 70)
    print("                    NETWORK ELASTICITY REGRESSION")
    print("                 Model: log(Stock_t) ~ log(Stock_t-1)")
    print("=" * 70)

    print("\n--- Model Summary ---")
    print(f"{'R':>20}: {np.sqrt(r_squared):>12.6f}")
    print(f"{'R Square':>20}: {r_squared:>12.6f}")
    print(f"{'Adjusted R Square':>20}: {r_squared_adj:>12.6f}")
    print(f"{'Std. Error of Est.':>20}: {rmse:>12.6f}")

    print("\n--- ANOVA ---")
    print(f"{'Source':<15} {'SS':>12} {'df':>6} {'MS':>12} {'F':>10} {'Sig.':>10}")
    print("-" * 65)
    print(
        f"{'Regression':<15} {ss_model:>12.4f} {df_model:>6} {ss_model/df_model:>12.4f} {f_stat:>10.4f} {f_pvalue:>10.6f}"
    )
    print(f"{'Residual':<15} {ss_residual:>12.4f} {df_residual:>6} {mse:>12.4f}")
    print(f"{'Total':<15} {ss_total:>12.4f} {df_total:>6}")

    print("\n--- Coefficients ---")
    print(
        f"{'Variable':<20} {'B':>10} {'SE':>10} {'t':>10} {'Sig.':>10} {'95% CI Lower':>12} {'95% CI Upper':>12}"
    )
    print("-" * 94)
    print(
        f"{'(Constant)':<20} {beta_hat[0]:>10.6f} {se_beta[0]:>10.6f} {t_stats[0]:>10.4f} {p_values[0]:>10.6f} {ci_lower[0]:>12.6f} {ci_upper[0]:>12.6f}"
    )
    print(
        f"{'log_stock_lag1':<20} {beta_hat[1]:>10.6f} {se_beta[1]:>10.6f} {t_stats[1]:>10.4f} {p_values[1]:>10.6f} {ci_lower[1]:>12.6f} {ci_upper[1]:>12.6f}"
    )

    print("\n--- Interpretation ---")
    print(f"Network Elasticity (beta): {beta_hat[1]:.4f}")
    if beta_hat[1] > 1:
        print(
            "  -> Accelerating growth: Larger diaspora communities grow proportionally faster"
        )
    elif beta_hat[1] < 1:
        print(
            "  -> Decelerating growth: Smaller diaspora communities grow proportionally faster"
        )
    else:
        print("  -> Proportional growth: All communities grow at similar rates")

    if p_values[1] < 0.001:
        sig_level = "***"
    elif p_values[1] < 0.01:
        sig_level = "**"
    elif p_values[1] < 0.05:
        sig_level = "*"
    else:
        sig_level = "ns"

    print(f"  Statistical significance: p = {p_values[1]:.6f} {sig_level}")
    print(
        f"  A 1% increase in lagged stock predicts a {beta_hat[1]:.4f}% increase in current stock"
    )

    # Store results
    results = {
        "model_specification": "log(Stock_t) = alpha + beta * log(Stock_t-1) + epsilon",
        "dependent_variable": "log(foreign_born_pop)",
        "independent_variable": "log(foreign_born_pop_lag1)",
        "sample_size": int(n),
        "degrees_of_freedom": {
            "model": int(df_model),
            "residual": int(df_residual),
            "total": int(df_total),
        },
        "coefficients": {
            "intercept": {
                "estimate": float(beta_hat[0]),
                "std_error": float(se_beta[0]),
                "t_value": float(t_stats[0]),
                "p_value": float(p_values[0]),
                "ci_95_lower": float(ci_lower[0]),
                "ci_95_upper": float(ci_upper[0]),
            },
            "network_elasticity": {
                "estimate": float(beta_hat[1]),
                "std_error": float(se_beta[1]),
                "t_value": float(t_stats[1]),
                "p_value": float(p_values[1]),
                "ci_95_lower": float(ci_lower[1]),
                "ci_95_upper": float(ci_upper[1]),
                "significance": sig_level,
            },
        },
        "model_fit": {
            "r_squared": float(r_squared),
            "r_squared_adjusted": float(r_squared_adj),
            "rmse": float(rmse),
            "correlation": float(correlation),
        },
        "anova": {
            "f_statistic": float(f_stat),
            "f_p_value": float(f_pvalue),
            "ss_regression": float(ss_model),
            "ss_residual": float(ss_residual),
            "ss_total": float(ss_total),
        },
        "interpretation": {
            "elasticity_type": "decelerating"
            if beta_hat[1] < 1
            else ("accelerating" if beta_hat[1] > 1 else "proportional"),
            "description": f"A 1% increase in lagged stock predicts a {beta_hat[1]:.4f}% increase in current stock",
        },
    }

    return results, nd_panel


def analyze_origin_growth(acs_df):
    """Analyze origin-specific growth rates in North Dakota."""
    print("\n[4] Analyzing Origin-Specific Growth...")
    print("-" * 70)

    # Filter to ND, country level
    nd_countries = acs_df[
        (acs_df["state_name"] == "North Dakota") & (acs_df["level"] == "country")
    ].copy()

    # Calculate year-over-year growth by origin
    nd_countries = nd_countries.sort_values(["country", "year"])
    nd_countries["pop_lag1"] = nd_countries.groupby("country")[
        "foreign_born_pop"
    ].shift(1)
    nd_countries["yoy_growth"] = (
        (nd_countries["foreign_born_pop"] - nd_countries["pop_lag1"])
        / nd_countries["pop_lag1"]
        * 100
    )

    # Get latest year data (2023)
    latest_year = nd_countries["year"].max()
    earliest_year = nd_countries["year"].min()

    # Calculate overall growth from first to last year
    first_year_data = nd_countries[nd_countries["year"] == earliest_year][
        ["country", "foreign_born_pop"]
    ].copy()
    first_year_data.columns = ["country", "pop_earliest"]

    last_year_data = nd_countries[nd_countries["year"] == latest_year][
        ["country", "foreign_born_pop"]
    ].copy()
    last_year_data.columns = ["country", "pop_latest"]

    growth_summary = first_year_data.merge(last_year_data, on="country", how="outer")
    growth_summary = growth_summary.dropna()

    # Filter to countries with meaningful population
    growth_summary = growth_summary[
        growth_summary["pop_earliest"] >= MIN_POPULATION_THRESHOLD
    ]

    # Calculate total growth
    growth_summary["absolute_growth"] = (
        growth_summary["pop_latest"] - growth_summary["pop_earliest"]
    )
    growth_summary["pct_growth"] = (
        growth_summary["absolute_growth"] / growth_summary["pop_earliest"]
    ) * 100
    years_span = latest_year - earliest_year
    growth_summary["cagr"] = (
        (growth_summary["pop_latest"] / growth_summary["pop_earliest"])
        ** (1 / years_span)
        - 1
    ) * 100

    # Sort by percentage growth
    growth_summary = growth_summary.sort_values("pct_growth", ascending=False)

    print(f"\nOrigin-Specific Growth Analysis ({earliest_year}-{latest_year})")
    print(
        f"Countries with initial population >= {MIN_POPULATION_THRESHOLD}: {len(growth_summary)}"
    )

    print("\n--- Top 15 Fastest Growing Origins (by % growth) ---")
    print(
        f"{'Country':<35} {'Pop '+str(earliest_year):>12} {'Pop '+str(latest_year):>12} {'% Growth':>12} {'CAGR %':>10}"
    )
    print("-" * 85)

    for _, row in growth_summary.head(15).iterrows():
        print(
            f"{row['country']:<35} {row['pop_earliest']:>12,.0f} {row['pop_latest']:>12,.0f} {row['pct_growth']:>12.1f} {row['cagr']:>10.2f}"
        )

    print("\n--- Bottom 5 Origins (declining or slow growth) ---")
    for _, row in growth_summary.tail(5).iterrows():
        print(
            f"{row['country']:<35} {row['pop_earliest']:>12,.0f} {row['pop_latest']:>12,.0f} {row['pct_growth']:>12.1f} {row['cagr']:>10.2f}"
        )

    # Calculate average YoY growth rates
    avg_yoy = (
        nd_countries.groupby("country")["yoy_growth"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    avg_yoy.columns = ["country", "mean_yoy_growth", "std_yoy_growth", "n_years"]
    avg_yoy = avg_yoy[avg_yoy["n_years"] >= 5]  # At least 5 years of data
    avg_yoy = avg_yoy.sort_values("mean_yoy_growth", ascending=False)

    print("\n--- Average Year-over-Year Growth Rates ---")
    print(f"{'Country':<35} {'Mean YoY %':>12} {'Std Dev':>12} {'N Years':>10}")
    print("-" * 75)
    for _, row in avg_yoy.head(10).iterrows():
        print(
            f"{row['country']:<35} {row['mean_yoy_growth']:>12.2f} {row['std_yoy_growth']:>12.2f} {row['n_years']:>10.0f}"
        )

    # Descriptive statistics for growth rates
    print("\n--- Descriptive Statistics: Growth Rates Across All Origins ---")
    print(f"{'Statistic':<25} {'Total Growth %':>15} {'CAGR %':>15}")
    print("-" * 60)
    print(f"{'N':.<25} {len(growth_summary):>15.0f} {len(growth_summary):>15.0f}")
    print(
        f"{'Mean':.<25} {growth_summary['pct_growth'].mean():>15.2f} {growth_summary['cagr'].mean():>15.2f}"
    )
    print(
        f"{'Std. Deviation':.<25} {growth_summary['pct_growth'].std():>15.2f} {growth_summary['cagr'].std():>15.2f}"
    )
    print(
        f"{'Minimum':.<25} {growth_summary['pct_growth'].min():>15.2f} {growth_summary['cagr'].min():>15.2f}"
    )
    print(
        f"{'25th Percentile':.<25} {growth_summary['pct_growth'].quantile(0.25):>15.2f} {growth_summary['cagr'].quantile(0.25):>15.2f}"
    )
    print(
        f"{'Median':.<25} {growth_summary['pct_growth'].median():>15.2f} {growth_summary['cagr'].median():>15.2f}"
    )
    print(
        f"{'75th Percentile':.<25} {growth_summary['pct_growth'].quantile(0.75):>15.2f} {growth_summary['cagr'].quantile(0.75):>15.2f}"
    )
    print(
        f"{'Maximum':.<25} {growth_summary['pct_growth'].max():>15.2f} {growth_summary['cagr'].max():>15.2f}"
    )

    results = {
        "period": f"{earliest_year}-{latest_year}",
        "n_countries_analyzed": int(len(growth_summary)),
        "min_population_filter": MIN_POPULATION_THRESHOLD,
        "top_growing_origins": growth_summary.head(15)[
            ["country", "pop_earliest", "pop_latest", "pct_growth", "cagr"]
        ].to_dict("records"),
        "declining_origins": growth_summary[growth_summary["pct_growth"] < 0][
            ["country", "pop_earliest", "pop_latest", "pct_growth", "cagr"]
        ].to_dict("records"),
        "descriptive_statistics": {
            "pct_growth": {
                "mean": float(growth_summary["pct_growth"].mean()),
                "std": float(growth_summary["pct_growth"].std()),
                "min": float(growth_summary["pct_growth"].min()),
                "q1": float(growth_summary["pct_growth"].quantile(0.25)),
                "median": float(growth_summary["pct_growth"].median()),
                "q3": float(growth_summary["pct_growth"].quantile(0.75)),
                "max": float(growth_summary["pct_growth"].max()),
            },
            "cagr": {
                "mean": float(growth_summary["cagr"].mean()),
                "std": float(growth_summary["cagr"].std()),
                "min": float(growth_summary["cagr"].min()),
                "q1": float(growth_summary["cagr"].quantile(0.25)),
                "median": float(growth_summary["cagr"].median()),
                "q3": float(growth_summary["cagr"].quantile(0.75)),
                "max": float(growth_summary["cagr"].max()),
            },
        },
    }

    return results, growth_summary


def analyze_diaspora_concentration(acs_df):
    """Analyze diaspora concentration - largest communities in ND."""
    print("\n[5] Analyzing Diaspora Concentration...")
    print("-" * 70)

    # Get most recent year data for ND at country level
    nd_countries = acs_df[
        (acs_df["state_name"] == "North Dakota") & (acs_df["level"] == "country")
    ].copy()

    latest_year = nd_countries["year"].max()
    latest_data = nd_countries[nd_countries["year"] == latest_year].copy()
    latest_data = latest_data.sort_values("foreign_born_pop", ascending=False)

    # Get total foreign-born for ND
    nd_total = acs_df[
        (acs_df["state_name"] == "North Dakota")
        & (acs_df["level"] == "total")
        & (acs_df["year"] == latest_year)
    ]["foreign_born_pop"].iloc[0]

    latest_data["pct_of_nd_foreign_born"] = (
        latest_data["foreign_born_pop"] / nd_total
    ) * 100
    latest_data["cumulative_pct"] = latest_data["pct_of_nd_foreign_born"].cumsum()

    print(f"\nDiaspora Concentration in North Dakota ({latest_year})")
    print(f"Total Foreign-Born Population: {nd_total:,.0f}")

    print("\n--- Top 20 Origin Countries ---")
    print(
        f"{'Rank':<6} {'Country':<35} {'Population':>12} {'MOE':>10} {'% of Total':>12} {'Cumulative %':>12}"
    )
    print("-" * 95)

    for rank, (_, row) in enumerate(latest_data.head(20).iterrows(), 1):
        print(
            f"{rank:<6} {row['country']:<35} {row['foreign_born_pop']:>12,.0f} {row['margin_of_error']:>10,.0f} {row['pct_of_nd_foreign_born']:>12.2f} {row['cumulative_pct']:>12.2f}"
        )

    # Herfindahl-Hirschman Index for concentration
    shares = latest_data["pct_of_nd_foreign_born"] / 100
    hhi = (shares**2).sum() * 10000  # Scale to 0-10000

    # Concentration ratio (top 5, top 10)
    cr5 = latest_data["pct_of_nd_foreign_born"].head(5).sum()
    cr10 = latest_data["pct_of_nd_foreign_born"].head(10).sum()

    print("\n--- Concentration Metrics ---")
    print(f"Herfindahl-Hirschman Index (HHI): {hhi:.2f}")
    print(
        f"  Interpretation: {'High concentration' if hhi > 2500 else 'Moderate concentration' if hhi > 1500 else 'Low concentration'}"
    )
    print(f"Top 5 Concentration Ratio (CR5): {cr5:.2f}%")
    print(f"Top 10 Concentration Ratio (CR10): {cr10:.2f}%")

    # Known high-concentration communities
    known_communities = [
        "Bhutan",
        "Nepal",
        "Somalia",
        "Philippines",
        "India",
        "China",
        "Mexico",
        "Canada",
    ]
    print("\n--- Known High-Concentration Communities ---")
    for country in known_communities:
        country_data = latest_data[
            latest_data["country"].str.contains(country, case=False, na=False)
        ]
        if len(country_data) > 0:
            row = country_data.iloc[0]
            print(
                f"  {row['country']}: {row['foreign_born_pop']:,.0f} ({row['pct_of_nd_foreign_born']:.2f}%)"
            )
        else:
            print(f"  {country}: Not found in data")

    results = {
        "year": int(latest_year),
        "total_foreign_born": float(nd_total),
        "n_origin_countries": int(len(latest_data)),
        "top_20_origins": latest_data.head(20)[
            ["country", "foreign_born_pop", "margin_of_error", "pct_of_nd_foreign_born"]
        ].to_dict("records"),
        "concentration_metrics": {
            "hhi": float(hhi),
            "hhi_interpretation": "high"
            if hhi > 2500
            else "moderate"
            if hhi > 1500
            else "low",
            "cr5": float(cr5),
            "cr10": float(cr10),
        },
    }

    return results, latest_data


def analyze_stock_flow_correlation(acs_df, dhs_df):
    """
    Analyze correlation between existing stock and new LPR admissions.
    Tests: Does larger diaspora predict more new immigrants?
    """
    print("\n[6] Analyzing Stock-Flow Correlation...")
    print("-" * 70)

    # Get ND data from both sources
    # ACS: foreign-born stock by country for 2023
    nd_acs = acs_df[
        (acs_df["state_name"] == "North Dakota")
        & (acs_df["level"] == "country")
        & (acs_df["year"] == 2023)
    ][["country", "foreign_born_pop", "margin_of_error"]].copy()
    nd_acs.columns = ["country", "acs_stock", "acs_moe"]

    # DHS: LPR admissions by country for FY2023
    nd_dhs = dhs_df[(dhs_df["state"] == "North Dakota") & (~dhs_df["is_region"])][
        ["region_country_of_birth", "lpr_count"]
    ].copy()
    nd_dhs.columns = ["country", "lpr_admissions"]

    print(f"ACS countries: {len(nd_acs)}")
    print(f"DHS countries: {len(nd_dhs)}")

    # Harmonize country names (basic matching)
    # Create mapping for known differences
    country_mapping = {
        "China": "China, People's Republic",
        "Korea": "Korea, South",
        "United Kingdom (inc. Crown Dependencies)": "United Kingdom",
        "Czechoslovakia (includes Czech Republic and Slovakia)": "Czech Republic",
        "Burma": "Burma (Myanmar)",
    }

    # Apply mapping to ACS data
    nd_acs["country_harmonized"] = nd_acs["country"].replace(country_mapping)
    nd_dhs["country_harmonized"] = nd_dhs["country"]

    # Merge on harmonized country names
    merged = nd_acs.merge(
        nd_dhs, on="country_harmonized", how="inner", suffixes=("_acs", "_dhs")
    )

    # Also try exact match for remaining countries not yet matched
    unmatched_acs = nd_acs[~nd_acs["country"].isin(merged["country_acs"])]
    unmatched_dhs = nd_dhs[~nd_dhs["country"].isin(merged["country_dhs"])]

    if len(unmatched_acs) > 0 and len(unmatched_dhs) > 0:
        exact_match = unmatched_acs.merge(
            unmatched_dhs,
            left_on="country",
            right_on="country",
            how="inner",
            suffixes=("_acs", "_dhs"),
        )
        if len(exact_match) > 0:
            # Standardize column names to match merged
            exact_match["country_acs"] = exact_match["country"]
            exact_match["country_dhs"] = exact_match["country"]
            merged = pd.concat([merged, exact_match], ignore_index=True)

    print(f"Matched countries: {len(merged)}")

    # Filter to countries with positive values
    merged = merged[(merged["acs_stock"] > 0) & (merged["lpr_admissions"] > 0)]
    print(f"Countries with positive stock and flow: {len(merged)}")

    if len(merged) < 5:
        print("WARNING: Too few matched countries for meaningful analysis")
        return {"error": "Too few matched countries", "n_matched": len(merged)}, None

    # Calculate correlations
    # Pearson correlation (linear relationship)
    pearson_r, pearson_p = stats.pearsonr(merged["acs_stock"], merged["lpr_admissions"])

    # Spearman correlation (monotonic relationship, robust to outliers)
    spearman_r, spearman_p = stats.spearmanr(
        merged["acs_stock"], merged["lpr_admissions"]
    )

    # Log-transformed correlation (network effect is often multiplicative)
    merged["log_stock"] = np.log(merged["acs_stock"] + 1)
    merged["log_lpr"] = np.log(merged["lpr_admissions"] + 1)
    log_pearson_r, log_pearson_p = stats.pearsonr(
        merged["log_stock"], merged["log_lpr"]
    )

    print("\n" + "=" * 70)
    print("                    STOCK-FLOW CORRELATION ANALYSIS")
    print("=" * 70)

    print("\n--- Correlation Matrix ---")
    print(f"{'Measure':<25} {'Correlation':>15} {'p-value':>15} {'95% CI':>25}")
    print("-" * 80)

    # Calculate CI for Pearson using Fisher transformation
    n = len(merged)
    z = np.arctanh(pearson_r)
    se = 1 / np.sqrt(n - 3)
    z_ci = stats.norm.ppf(0.975) * se
    ci_low = np.tanh(z - z_ci)
    ci_high = np.tanh(z + z_ci)

    print(
        f"{'Pearson r (levels)':<25} {pearson_r:>15.4f} {pearson_p:>15.6f} [{ci_low:.4f}, {ci_high:.4f}]"
    )
    print(f"{'Spearman rho (ranks)':<25} {spearman_r:>15.4f} {spearman_p:>15.6f}")

    # CI for log correlation
    z_log = np.arctanh(log_pearson_r)
    ci_low_log = np.tanh(z_log - z_ci)
    ci_high_log = np.tanh(z_log + z_ci)
    print(
        f"{'Pearson r (log-log)':<25} {log_pearson_r:>15.4f} {log_pearson_p:>15.6f} [{ci_low_log:.4f}, {ci_high_log:.4f}]"
    )

    print("\n--- Interpretation ---")
    if pearson_r > 0.7:
        strength = "Strong positive"
    elif pearson_r > 0.4:
        strength = "Moderate positive"
    elif pearson_r > 0.2:
        strength = "Weak positive"
    elif pearson_r > -0.2:
        strength = "No meaningful"
    else:
        strength = "Negative"

    print(
        f"Relationship: {strength} correlation between existing diaspora size and new admissions"
    )
    print(
        f"Statistical significance: p = {pearson_p:.6f} ({'***' if pearson_p < 0.001 else '**' if pearson_p < 0.01 else '*' if pearson_p < 0.05 else 'ns'})"
    )

    if pearson_r > 0.2 and pearson_p < 0.05:
        print(
            "CONCLUSION: Evidence supports network effects - larger diasporas attract more immigrants"
        )
    else:
        print(
            "CONCLUSION: Limited evidence for network effects in this cross-sectional data"
        )

    print("\n--- Top Countries by Stock and Flow ---")
    print(f"{'Country':<30} {'Stock (ACS)':>12} {'Flow (LPR)':>12} {'Ratio':>10}")
    print("-" * 70)
    merged["ratio"] = merged["lpr_admissions"] / merged["acs_stock"] * 1000

    # Determine which country column to use for display
    country_col = (
        "country_harmonized"
        if "country_harmonized" in merged.columns
        else "country_acs"
    )
    for _, row in merged.nlargest(10, "acs_stock").iterrows():
        country_name = row.get(country_col, row.get("country", "N/A"))
        print(
            f"{country_name:<30} {row['acs_stock']:>12,.0f} {row['lpr_admissions']:>12,.0f} {row['ratio']:>10.2f}"
        )

    # Prepare matched countries for export
    export_cols = ["acs_stock", "lpr_admissions"]
    if "country_harmonized" in merged.columns:
        export_cols = ["country_harmonized"] + export_cols
    elif "country_acs" in merged.columns:
        merged["country_harmonized"] = merged["country_acs"]
        export_cols = ["country_harmonized"] + export_cols

    results = {
        "n_countries_matched": int(len(merged)),
        "correlation_analysis": {
            "pearson": {
                "r": float(pearson_r),
                "p_value": float(pearson_p),
                "ci_95_lower": float(ci_low),
                "ci_95_upper": float(ci_high),
            },
            "spearman": {"rho": float(spearman_r), "p_value": float(spearman_p)},
            "log_pearson": {
                "r": float(log_pearson_r),
                "p_value": float(log_pearson_p),
                "ci_95_lower": float(ci_low_log),
                "ci_95_upper": float(ci_high_log),
            },
        },
        "interpretation": {
            "strength": strength,
            "network_effect_support": bool(pearson_r > 0.2 and pearson_p < 0.05),
        },
        "matched_countries": merged[export_cols].to_dict("records"),
    }

    return results, merged


def create_figures(nd_panel, growth_summary, diaspora_data, stock_flow_data):
    """Create all required figures."""
    print("\n[7] Creating Figures...")
    print("-" * 70)

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: Network Elasticity Scatter
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    ax1.scatter(
        nd_panel["log_stock_lag1"],
        nd_panel["log_stock"],
        alpha=0.6,
        s=50,
        edgecolors="white",
        linewidth=0.5,
    )

    # Add regression line
    x = nd_panel["log_stock_lag1"].values
    y = nd_panel["log_stock"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = intercept + slope * x_line
    ax1.plot(
        x_line,
        y_line,
        "r-",
        linewidth=2,
        label=f"OLS: y = {intercept:.3f} + {slope:.3f}x",
    )

    # Add 45-degree line for reference
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax1.plot(
        [min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="45-degree line"
    )

    ax1.set_xlabel("log(Foreign-Born Stock, t-1)", fontsize=12)
    ax1.set_ylabel("log(Foreign-Born Stock, t)", fontsize=12)
    ax1.set_title(
        f"Network Elasticity: Diaspora Persistence in North Dakota\n(R² = {r_value**2:.3f}, n = {len(nd_panel)})",
        fontsize=14,
    )
    ax1.legend(loc="lower right")

    # Add annotation
    ax1.text(
        0.05,
        0.95,
        f"Network Elasticity (β) = {slope:.4f}\nSE = {std_err:.4f}\np < 0.001",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()
    fig1.savefig(
        FIGURES_DIR / "module_3_2_network_elasticity.png", dpi=300, bbox_inches="tight"
    )
    fig1.savefig(
        FIGURES_DIR / "module_3_2_network_elasticity.pdf", dpi=300, bbox_inches="tight"
    )
    print("    Saved: module_3_2_network_elasticity.png/pdf")
    plt.close(fig1)

    # Figure 2: Top Origins Growth (bar chart)
    fig2, ax2 = plt.subplots(figsize=(12, 8))

    top_15 = growth_summary.head(15).copy()
    colors = ["#2ecc71" if x > 0 else "#e74c3c" for x in top_15["pct_growth"]]

    bars = ax2.barh(
        range(len(top_15)), top_15["pct_growth"], color=colors, edgecolor="white"
    )
    ax2.set_yticks(range(len(top_15)))
    ax2.set_yticklabels(top_15["country"])
    ax2.invert_yaxis()
    ax2.set_xlabel("Percentage Growth (2009-2023)", fontsize=12)
    ax2.set_title(
        "Top 15 Fastest Growing Immigrant Origins in North Dakota", fontsize=14
    )
    ax2.axvline(x=0, color="black", linewidth=0.5)

    # Add value labels
    for i, (_bar, val) in enumerate(zip(bars, top_15["pct_growth"], strict=False)):
        if val >= 0:
            ax2.text(val + 5, i, f"{val:.0f}%", va="center", fontsize=9)
        else:
            ax2.text(val - 20, i, f"{val:.0f}%", va="center", fontsize=9)

    plt.tight_layout()
    fig2.savefig(
        FIGURES_DIR / "module_3_2_top_origins_growth.png", dpi=300, bbox_inches="tight"
    )
    fig2.savefig(
        FIGURES_DIR / "module_3_2_top_origins_growth.pdf", dpi=300, bbox_inches="tight"
    )
    print("    Saved: module_3_2_top_origins_growth.png/pdf")
    plt.close(fig2)

    # Figure 3: Diaspora Sizes (top 15)
    fig3, ax3 = plt.subplots(figsize=(12, 8))

    top_diaspora = diaspora_data.head(15).copy()
    bars = ax3.barh(
        range(len(top_diaspora)),
        top_diaspora["foreign_born_pop"],
        color="steelblue",
        edgecolor="white",
    )

    # Add error bars for MOE
    ax3.errorbar(
        top_diaspora["foreign_born_pop"],
        range(len(top_diaspora)),
        xerr=top_diaspora["margin_of_error"],
        fmt="none",
        color="black",
        capsize=3,
        alpha=0.7,
    )

    ax3.set_yticks(range(len(top_diaspora)))
    ax3.set_yticklabels(top_diaspora["country"])
    ax3.invert_yaxis()
    ax3.set_xlabel("Foreign-Born Population (2023)", fontsize=12)
    ax3.set_title(
        "Top 15 Diaspora Communities in North Dakota (with 90% MOE)", fontsize=14
    )

    # Add value labels
    for i, (_bar, val, pct) in enumerate(
        zip(
            bars,
            top_diaspora["foreign_born_pop"],
            top_diaspora["pct_of_nd_foreign_born"],
            strict=False,
        )
    ):
        ax3.text(val + 100, i, f"{val:,.0f} ({pct:.1f}%)", va="center", fontsize=9)

    plt.tight_layout()
    fig3.savefig(
        FIGURES_DIR / "module_3_2_diaspora_sizes.png", dpi=300, bbox_inches="tight"
    )
    fig3.savefig(
        FIGURES_DIR / "module_3_2_diaspora_sizes.pdf", dpi=300, bbox_inches="tight"
    )
    print("    Saved: module_3_2_diaspora_sizes.png/pdf")
    plt.close(fig3)

    # Figure 4: Stock-Flow Correlation
    if stock_flow_data is not None and len(stock_flow_data) >= 5:
        fig4, ax4 = plt.subplots(figsize=(10, 8))

        ax4.scatter(
            stock_flow_data["acs_stock"],
            stock_flow_data["lpr_admissions"],
            alpha=0.7,
            s=80,
            edgecolors="white",
            linewidth=0.5,
        )

        # Add regression line
        x = stock_flow_data["acs_stock"].values
        y = stock_flow_data["lpr_admissions"].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = intercept + slope * x_line
        ax4.plot(x_line, y_line, "r-", linewidth=2, label=f"OLS: r = {r_value:.3f}")

        # Label some points
        top_countries = stock_flow_data.nlargest(5, "acs_stock")
        for _, row in top_countries.iterrows():
            # Try different possible column names for country
            country_name = ""
            for col in ["country", "country_acs", "country_harmonized"]:
                if col in row.index and pd.notna(row[col]):
                    country_name = str(row[col])
                    break
            # Truncate long names
            if len(country_name) > 15:
                country_name = country_name[:12] + "..."
            ax4.annotate(
                country_name,
                (row["acs_stock"], row["lpr_admissions"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        ax4.set_xlabel("Existing Foreign-Born Stock (ACS 2023)", fontsize=12)
        ax4.set_ylabel("New LPR Admissions (FY2023)", fontsize=12)
        ax4.set_title(
            f"Stock-Flow Correlation: Diaspora Size vs. New Arrivals\n(n = {len(stock_flow_data)}, r = {r_value:.3f}, p = {p_value:.4f})",
            fontsize=14,
        )
        ax4.legend(loc="upper left")

        plt.tight_layout()
        fig4.savefig(
            FIGURES_DIR / "module_3_2_stock_flow_correlation.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig4.savefig(
            FIGURES_DIR / "module_3_2_stock_flow_correlation.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        print("    Saved: module_3_2_stock_flow_correlation.png/pdf")
        plt.close(fig4)
    else:
        print("    WARNING: Insufficient data for stock-flow correlation figure")


def save_results(
    elasticity_results,
    origin_results,
    diaspora_results,
    stock_flow_results,
    panel_decisions,
    nd_panel,
):
    """Save all results to output files."""
    print("\n[8] Saving Results...")
    print("-" * 70)

    # Combine elasticity results with decisions
    elasticity_output = {
        "analysis_type": "Network Elasticity Estimation",
        "module": "3.2",
        "analysis_date": datetime.now().isoformat(),
        "data_preparation": panel_decisions,
        "regression_results": elasticity_results,
        "limitations": [
            "LPR state x country data only available for FY2023",
            "ACS has sampling error - margin of error columns included",
            "Small counts for some origins may be unreliable (filtered < 50)",
            "Log transformation required adding constant to handle zeros",
        ],
    }

    with open(OUTPUT_DIR / "module_3_2_network_elasticity.json", "w") as f:
        json.dump(elasticity_output, f, indent=2)
    print("    Saved: module_3_2_network_elasticity.json")

    # Origin analysis results
    origin_output = {
        "analysis_type": "Origin-Specific Panel Analysis",
        "module": "3.2",
        "analysis_date": datetime.now().isoformat(),
        "growth_analysis": origin_results,
        "diaspora_concentration": diaspora_results,
        "stock_flow_correlation": stock_flow_results,
    }

    with open(OUTPUT_DIR / "module_3_2_origin_analysis.json", "w") as f:
        json.dump(origin_output, f, indent=2)
    print("    Saved: module_3_2_origin_analysis.json")

    # Save panel data as parquet
    nd_panel.to_parquet(OUTPUT_DIR / "module_3_2_origin_panel.parquet", index=False)
    print("    Saved: module_3_2_origin_panel.parquet")


def main():
    """Main execution function."""
    # Load data
    acs_state_origin, acs_nd_share, dhs_lpr = load_data()

    # Prepare panel data
    nd_panel, full_panel, panel_decisions = prepare_panel_data(acs_state_origin)

    # Estimate network elasticity
    elasticity_results, nd_panel_updated = estimate_network_elasticity(nd_panel)

    # Analyze origin-specific growth
    origin_results, growth_summary = analyze_origin_growth(acs_state_origin)

    # Analyze diaspora concentration
    diaspora_results, diaspora_data = analyze_diaspora_concentration(acs_state_origin)

    # Analyze stock-flow correlation
    stock_flow_results, stock_flow_data = analyze_stock_flow_correlation(
        acs_state_origin, dhs_lpr
    )

    # Create figures
    create_figures(nd_panel, growth_summary, diaspora_data, stock_flow_data)

    # Save results
    save_results(
        elasticity_results,
        origin_results,
        diaspora_results,
        stock_flow_results,
        panel_decisions,
        nd_panel,
    )

    print("\n" + "=" * 70)
    print("MODULE 3.2 COMPLETE")
    print("=" * 70)
    print("\nOutputs saved to:")
    print(f"  Results: {OUTPUT_DIR}")
    print(f"  Figures: {FIGURES_DIR}")

    return {
        "elasticity": elasticity_results,
        "origins": origin_results,
        "diaspora": diaspora_results,
        "stock_flow": stock_flow_results,
    }


if __name__ == "__main__":
    results = main()
