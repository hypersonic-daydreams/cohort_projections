#!/usr/bin/env python3
"""
Module 7 Robustness: Wild Cluster Bootstrap and Restricted Pre-Period DiD
==========================================================================

Implements causal inference robustness checks for P3.10 and P3.14:
1. Wild cluster bootstrap for DiD ATT (small-sample inference with 7 treated clusters)
2. Randomization/permutation inference as alternative
3. Restricted pre-period DiD (using years with more parallel trends)

Usage:
    python module_7_robustness.py
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"
RESULTS_DIR = Path(__file__).parent / "results"
ARTICLE_DIR = Path(__file__).parent / "journal_article"

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)


def load_refugee_data() -> pd.DataFrame:
    """Load refugee arrivals data."""
    refugee_path = DATA_DIR / "refugee_arrivals_by_state_nationality.parquet"
    df_refugee = pd.read_parquet(refugee_path)
    print(f"Loaded refugee arrivals: {df_refugee.shape}")
    return df_refugee


def prepare_travel_ban_did_data(df_refugee: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for Travel Ban DiD analysis.

    Treatment: Nationalities affected by Travel Ban
    Event year: 2017 (first full post-treatment year: 2018)
    """
    travel_ban_countries = [
        "Iran",
        "Iraq",
        "Libya",
        "Somalia",
        "Sudan",
        "Syria",
        "Yemen",
    ]

    # Aggregate refugee arrivals by nationality and year
    df_nat_year = (
        df_refugee.groupby(["fiscal_year", "nationality"])["arrivals"]
        .sum()
        .reset_index()
    )
    df_nat_year.columns = ["year", "nationality", "arrivals"]

    # Create treatment indicator
    df_nat_year["treated"] = (
        df_nat_year["nationality"].isin(travel_ban_countries).astype(int)
    )

    # Post indicator (2018 is first full post-treatment year)
    df_nat_year["post"] = (df_nat_year["year"] >= 2018).astype(int)

    # Interaction term (DiD estimator)
    df_nat_year["treated_x_post"] = df_nat_year["treated"] * df_nat_year["post"]

    # Log arrivals (add 1 to handle zeros)
    df_nat_year["log_arrivals"] = np.log(df_nat_year["arrivals"] + 1)

    # Relative time to treatment
    df_nat_year["rel_time"] = df_nat_year["year"] - 2018

    return df_nat_year


def estimate_did_twfe(df: pd.DataFrame, pre_period_start: int | None = None) -> dict:
    """
    Estimate TWFE DiD model, optionally restricting pre-period.

    Args:
        df: Prepared DiD data
        pre_period_start: If provided, restrict pre-period to years >= this value

    Returns:
        Dictionary with estimation results
    """
    # Exclude 2020 due to COVID confounding
    df_analysis = df[df["year"] < 2020].copy()

    # Optionally restrict pre-period
    if pre_period_start is not None:
        df_analysis = df_analysis[
            (df_analysis["year"] >= pre_period_start) | (df_analysis["post"] == 1)
        ].copy()
        pre_period_label = f"{pre_period_start}-2017"
    else:
        pre_period_label = "2002-2017"

    # Reset index
    df_analysis = df_analysis.reset_index(drop=True)

    # Sample info
    n_treated = df_analysis[df_analysis["treated"] == 1]["nationality"].nunique()
    n_control = df_analysis[df_analysis["treated"] == 0]["nationality"].nunique()
    n_pre = len(df_analysis[df_analysis["post"] == 0]["year"].unique())
    n_post = len(df_analysis[df_analysis["post"] == 1]["year"].unique())

    # Design matrix
    y = df_analysis["log_arrivals"].values
    X = pd.DataFrame({"treated_x_post": df_analysis["treated_x_post"].values})

    # Add nationality fixed effects
    nationality_dummies = pd.get_dummies(
        df_analysis["nationality"], prefix="nat", drop_first=True
    )
    X_fe = pd.concat(
        [X.reset_index(drop=True), nationality_dummies.reset_index(drop=True)], axis=1
    )

    # Add year fixed effects
    year_dummies = pd.get_dummies(df_analysis["year"], prefix="year", drop_first=True)
    X_twfe = pd.concat(
        [X_fe.reset_index(drop=True), year_dummies.reset_index(drop=True)], axis=1
    )

    # Add constant
    X_twfe = sm.add_constant(X_twfe)
    X_twfe = X_twfe.astype(float)

    # Estimate with clustered standard errors
    model_twfe = OLS(y, X_twfe).fit(
        cov_type="cluster", cov_kwds={"groups": df_analysis["nationality"]}
    )

    # Extract ATT
    att = model_twfe.params["treated_x_post"]
    se = model_twfe.bse["treated_x_post"]
    t_stat = model_twfe.tvalues["treated_x_post"]
    p_value = model_twfe.pvalues["treated_x_post"]
    ci = model_twfe.conf_int().loc["treated_x_post"]

    return {
        "att": float(att),
        "se": float(se),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "n_treated": n_treated,
        "n_control": n_control,
        "n_pre": n_pre,
        "n_post": n_post,
        "n_obs": len(df_analysis),
        "pre_period": pre_period_label,
        "df_analysis": df_analysis,
        "model": model_twfe,
        "y": y,
        "X": X_twfe,
    }


def wild_cluster_bootstrap(
    df: pd.DataFrame,
    n_bootstrap: int = 9999,
    seed: int = 42,
    pre_period_start: int | None = None,
) -> dict:
    """
    Implement wild cluster bootstrap for DiD with Rademacher weights.

    Following Cameron, Gelbach, and Miller (2008) approach for few clusters.

    Args:
        df: Prepared DiD data
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed
        pre_period_start: If provided, restrict pre-period

    Returns:
        Dictionary with bootstrap results
    """
    print("\n" + "=" * 60)
    print("WILD CLUSTER BOOTSTRAP FOR DiD ATT")
    print("=" * 60)

    np.random.seed(seed)

    # Get base estimation
    base_results = estimate_did_twfe(df, pre_period_start)
    df_analysis = base_results["df_analysis"]
    att_original = base_results["att"]

    print(f"\nOriginal ATT: {att_original:.4f}")
    print(f"Original SE (clustered): {base_results['se']:.4f}")
    print(f"Original p-value: {base_results['p_value']:.4f}")

    # Get unique clusters (nationalities)
    clusters = df_analysis["nationality"].unique()
    n_clusters = len(clusters)
    n_treated_clusters = df_analysis[df_analysis["treated"] == 1][
        "nationality"
    ].nunique()

    print(f"\nNumber of clusters: {n_clusters}")
    print(f"Number of treated clusters: {n_treated_clusters}")
    print(f"Bootstrap iterations: {n_bootstrap}")

    # Create cluster mapping
    cluster_map = {c: i for i, c in enumerate(clusters)}
    df_analysis["cluster_id"] = df_analysis["nationality"].map(cluster_map)

    # Prepare data for bootstrap
    y = df_analysis["log_arrivals"].values.copy()

    # Create design matrix without the treatment effect variable
    # (for computing residuals under H0)
    X_h0 = pd.DataFrame()

    # Add nationality fixed effects
    nationality_dummies = pd.get_dummies(
        df_analysis["nationality"], prefix="nat", drop_first=True
    )
    X_h0 = pd.concat([X_h0, nationality_dummies.reset_index(drop=True)], axis=1)

    # Add year fixed effects
    year_dummies = pd.get_dummies(df_analysis["year"], prefix="year", drop_first=True)
    X_h0 = pd.concat([X_h0, year_dummies.reset_index(drop=True)], axis=1)

    X_h0 = sm.add_constant(X_h0)
    X_h0 = X_h0.astype(float)

    # Fit restricted model (under H0: no treatment effect)
    model_h0 = OLS(y, X_h0).fit()
    residuals_h0 = model_h0.resid

    # Full design matrix (with treatment effect)
    X_full = pd.DataFrame({"treated_x_post": df_analysis["treated_x_post"].values})
    X_full = pd.concat(
        [X_full.reset_index(drop=True), nationality_dummies.reset_index(drop=True)],
        axis=1,
    )
    X_full = pd.concat(
        [X_full.reset_index(drop=True), year_dummies.reset_index(drop=True)], axis=1
    )
    X_full = sm.add_constant(X_full)
    X_full = X_full.astype(float)

    # Bootstrap loop
    bootstrap_t_stats = []

    for b in range(n_bootstrap):
        # Generate Rademacher weights at cluster level
        weights = np.random.choice([-1, 1], size=n_clusters)

        # Apply weights to residuals by cluster
        weighted_residuals = np.zeros_like(residuals_h0)
        for c_idx, c_name in enumerate(clusters):
            mask = df_analysis["nationality"] == c_name
            weighted_residuals[mask] = residuals_h0[mask] * weights[c_idx]

        # Create bootstrap outcome under H0
        y_boot = model_h0.fittedvalues + weighted_residuals

        # Estimate full model on bootstrap sample
        try:
            model_boot = OLS(y_boot, X_full).fit(
                cov_type="cluster", cov_kwds={"groups": df_analysis["nationality"]}
            )
            t_boot = model_boot.tvalues["treated_x_post"]
            bootstrap_t_stats.append(t_boot)
        except Exception:
            # Skip failed iterations
            continue

    bootstrap_t_stats = np.array(bootstrap_t_stats)
    n_valid = len(bootstrap_t_stats)

    # Calculate bootstrap p-value (two-sided)
    t_original = base_results["t_stat"]
    p_bootstrap = np.mean(np.abs(bootstrap_t_stats) >= np.abs(t_original))

    # Calculate bootstrap confidence interval
    # Using percentile method on t-statistics
    t_lower = np.percentile(bootstrap_t_stats, 2.5)
    t_upper = np.percentile(bootstrap_t_stats, 97.5)

    # Convert to coefficient CI
    ci_lower_boot = att_original - t_upper * base_results["se"]
    ci_upper_boot = att_original - t_lower * base_results["se"]

    print(f"\n{'='*60}")
    print("WILD CLUSTER BOOTSTRAP RESULTS")
    print("=" * 60)
    print(f"Valid bootstrap iterations: {n_valid}")
    print(f"\nOriginal t-statistic: {t_original:.4f}")
    print(
        f"Bootstrap t-statistics: mean={np.mean(bootstrap_t_stats):.4f}, "
        f"SD={np.std(bootstrap_t_stats):.4f}"
    )
    print(f"\nConventional p-value: {base_results['p_value']:.4f}")
    print(f"Bootstrap p-value: {p_bootstrap:.4f}")
    print(
        f"\nConventional 95% CI: [{base_results['ci_lower']:.4f}, "
        f"{base_results['ci_upper']:.4f}]"
    )
    print(f"Bootstrap 95% CI: [{ci_lower_boot:.4f}, {ci_upper_boot:.4f}]")

    return {
        "method": "Wild Cluster Bootstrap (Rademacher weights)",
        "n_clusters": n_clusters,
        "n_treated_clusters": n_treated_clusters,
        "n_bootstrap": n_bootstrap,
        "n_valid_iterations": n_valid,
        "original_att": att_original,
        "original_se": base_results["se"],
        "original_t_stat": t_original,
        "conventional_p_value": base_results["p_value"],
        "bootstrap_p_value": p_bootstrap,
        "conventional_ci_lower": base_results["ci_lower"],
        "conventional_ci_upper": base_results["ci_upper"],
        "bootstrap_ci_lower": ci_lower_boot,
        "bootstrap_ci_upper": ci_upper_boot,
        "bootstrap_t_mean": float(np.mean(bootstrap_t_stats)),
        "bootstrap_t_sd": float(np.std(bootstrap_t_stats)),
    }


def randomization_inference(
    df: pd.DataFrame,
    n_permutations: int = 9999,
    seed: int = 42,
    pre_period_start: int | None = None,
) -> dict:
    """
    Implement randomization (permutation) inference for DiD.

    Permutes treatment assignment across clusters to generate null distribution.

    Args:
        df: Prepared DiD data
        n_permutations: Number of permutations
        seed: Random seed
        pre_period_start: If provided, restrict pre-period

    Returns:
        Dictionary with permutation inference results
    """
    print("\n" + "=" * 60)
    print("RANDOMIZATION INFERENCE FOR DiD ATT")
    print("=" * 60)

    np.random.seed(seed)

    # Get base estimation
    base_results = estimate_did_twfe(df, pre_period_start)
    df_analysis = base_results["df_analysis"].copy()
    att_original = base_results["att"]

    print(f"\nOriginal ATT: {att_original:.4f}")
    print(f"Permutations: {n_permutations}")

    # Get cluster-level treatment assignment
    cluster_treatment = (
        df_analysis.groupby("nationality")["treated"].first().reset_index()
    )
    nationalities = cluster_treatment["nationality"].values
    original_treatment = cluster_treatment["treated"].values
    n_treated = original_treatment.sum()

    print(f"Treated clusters: {n_treated} of {len(nationalities)}")

    # Permutation loop
    permutation_atts = []

    for p in range(n_permutations):
        # Permute treatment assignment at cluster level
        perm_treatment = np.random.permutation(original_treatment)

        # Create mapping
        perm_map = dict(zip(nationalities, perm_treatment, strict=False))

        # Apply to data
        df_perm = df_analysis.copy()
        df_perm["treated_perm"] = df_perm["nationality"].map(perm_map)
        df_perm["treated_x_post_perm"] = df_perm["treated_perm"] * df_perm["post"]

        # Estimate DiD with permuted treatment
        y = df_perm["log_arrivals"].values

        X = pd.DataFrame({"treated_x_post": df_perm["treated_x_post_perm"].values})

        nationality_dummies = pd.get_dummies(
            df_perm["nationality"], prefix="nat", drop_first=True
        )
        X = pd.concat(
            [X.reset_index(drop=True), nationality_dummies.reset_index(drop=True)],
            axis=1,
        )

        year_dummies = pd.get_dummies(df_perm["year"], prefix="year", drop_first=True)
        X = pd.concat(
            [X.reset_index(drop=True), year_dummies.reset_index(drop=True)], axis=1
        )

        X = sm.add_constant(X)
        X = X.astype(float)

        try:
            model_perm = OLS(y, X).fit()
            att_perm = model_perm.params["treated_x_post"]
            permutation_atts.append(att_perm)
        except Exception:
            continue

    permutation_atts = np.array(permutation_atts)
    n_valid = len(permutation_atts)

    # Calculate permutation p-value (two-sided)
    p_perm = np.mean(np.abs(permutation_atts) >= np.abs(att_original))

    # Also calculate one-sided p-value (for negative effect)
    p_perm_onesided = np.mean(permutation_atts <= att_original)

    # Rank of original among permutations
    rank = np.sum(permutation_atts <= att_original) + 1

    print(f"\n{'='*60}")
    print("RANDOMIZATION INFERENCE RESULTS")
    print("=" * 60)
    print(f"Valid permutations: {n_valid}")
    print(f"\nOriginal ATT: {att_original:.4f}")
    print(
        f"Permutation ATTs: mean={np.mean(permutation_atts):.4f}, "
        f"SD={np.std(permutation_atts):.4f}"
    )
    print(
        f"Permutation ATTs: min={np.min(permutation_atts):.4f}, "
        f"max={np.max(permutation_atts):.4f}"
    )
    print(f"\nRank of original: {rank} / {n_valid + 1}")
    print(f"Two-sided p-value: {p_perm:.4f}")
    print(f"One-sided p-value (ATT <= observed): {p_perm_onesided:.4f}")

    return {
        "method": "Randomization Inference (Fisher exact test)",
        "n_permutations": n_permutations,
        "n_valid_permutations": n_valid,
        "n_treated_clusters": int(n_treated),
        "n_total_clusters": len(nationalities),
        "original_att": att_original,
        "permutation_att_mean": float(np.mean(permutation_atts)),
        "permutation_att_sd": float(np.std(permutation_atts)),
        "permutation_att_min": float(np.min(permutation_atts)),
        "permutation_att_max": float(np.max(permutation_atts)),
        "rank": int(rank),
        "p_value_twosided": p_perm,
        "p_value_onesided": p_perm_onesided,
    }


def restricted_preperiod_analysis(df: pd.DataFrame) -> dict:
    """
    Estimate DiD with restricted pre-periods to assess robustness.

    Based on event study showing pre-trends are more problematic in early years.

    Returns:
        Dictionary with results for different pre-period restrictions
    """
    print("\n" + "=" * 60)
    print("RESTRICTED PRE-PERIOD DiD ANALYSIS")
    print("=" * 60)

    results = {}

    # Full pre-period (baseline)
    print("\n--- Full Pre-Period (2002-2017) ---")
    full_results = estimate_did_twfe(df, pre_period_start=None)
    results["full_2002_2017"] = {
        "pre_period": "2002-2017",
        "n_pre_years": full_results["n_pre"],
        "att": full_results["att"],
        "se": full_results["se"],
        "t_stat": full_results["t_stat"],
        "p_value": full_results["p_value"],
        "ci_lower": full_results["ci_lower"],
        "ci_upper": full_results["ci_upper"],
        "n_obs": full_results["n_obs"],
    }
    print(f"ATT: {full_results['att']:.4f} (SE: {full_results['se']:.4f})")
    print(f"p-value: {full_results['p_value']:.4f}")

    # Restricted pre-periods
    restrictions = [
        (2010, "2010-2017 (8 years)"),
        (2012, "2012-2017 (6 years)"),
        (2013, "2013-2017 (5 years)"),
        (2014, "2014-2017 (4 years)"),
        (2015, "2015-2017 (3 years)"),
    ]

    for start_year, label in restrictions:
        print(f"\n--- Restricted Pre-Period ({label}) ---")
        restricted_results = estimate_did_twfe(df, pre_period_start=start_year)
        key = f"restricted_{start_year}_2017"
        results[key] = {
            "pre_period": label,
            "n_pre_years": restricted_results["n_pre"],
            "att": restricted_results["att"],
            "se": restricted_results["se"],
            "t_stat": restricted_results["t_stat"],
            "p_value": restricted_results["p_value"],
            "ci_lower": restricted_results["ci_lower"],
            "ci_upper": restricted_results["ci_upper"],
            "n_obs": restricted_results["n_obs"],
        }
        print(
            f"ATT: {restricted_results['att']:.4f} "
            f"(SE: {restricted_results['se']:.4f})"
        )
        print(f"p-value: {restricted_results['p_value']:.4f}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: RESTRICTED PRE-PERIOD SENSITIVITY")
    print("=" * 60)
    print(f"{'Pre-Period':<20} {'ATT':>10} {'SE':>10} {'p-value':>10} {'Sig':>6}")
    print("-" * 60)
    for key, res in results.items():
        sig = (
            "***"
            if res["p_value"] < 0.001
            else (
                "**"
                if res["p_value"] < 0.01
                else (
                    "*"
                    if res["p_value"] < 0.05
                    else ("+" if res["p_value"] < 0.10 else "")
                )
            )
        )
        print(
            f"{res['pre_period']:<20} {res['att']:>10.4f} {res['se']:>10.4f} "
            f"{res['p_value']:>10.4f} {sig:>6}"
        )

    return results


def run_joint_pretrend_test(
    df: pd.DataFrame, pre_period_start: int | None = None
) -> dict:
    """
    Run joint F-test for pre-trends with optional restricted pre-period.
    """
    # Exclude 2020 and optionally restrict pre-period
    df_analysis = df[df["year"] < 2020].copy()

    if pre_period_start is not None:
        df_analysis = df_analysis[
            (df_analysis["year"] >= pre_period_start) | (df_analysis["post"] == 1)
        ].copy()

    df_analysis = df_analysis.reset_index(drop=True)
    df_analysis["rel_time"] = df_analysis["year"] - 2018

    # Get unique relative times (excluding reference period -1)
    rel_times = sorted(df_analysis["rel_time"].unique())
    rel_times = [int(t) for t in rel_times]
    rel_times_no_ref = [t for t in rel_times if t != -1]
    pre_times = [t for t in rel_times_no_ref if t < 0]

    if len(pre_times) == 0:
        return {"f_statistic": np.nan, "p_value": np.nan, "n_pre_coefs": 0}

    # Create interaction dummies
    for t in rel_times_no_ref:
        df_analysis[f"treated_x_t{t}"] = (
            df_analysis["treated"] * (df_analysis["rel_time"] == t)
        ).astype(float)

    # Design matrix
    y = df_analysis["log_arrivals"].values
    interaction_cols = [f"treated_x_t{t}" for t in rel_times_no_ref]

    X = pd.DataFrame({col: df_analysis[col].values for col in interaction_cols})

    nat_dummies = pd.get_dummies(
        df_analysis["nationality"], prefix="nat", drop_first=True
    )
    X = pd.concat(
        [X.reset_index(drop=True), nat_dummies.reset_index(drop=True)], axis=1
    )

    year_dummies = pd.get_dummies(df_analysis["year"], prefix="year", drop_first=True)
    X = pd.concat(
        [X.reset_index(drop=True), year_dummies.reset_index(drop=True)], axis=1
    )

    X = sm.add_constant(X)
    X = X.astype(float)

    es_model = OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": df_analysis["nationality"]}
    )

    # Pre-trend F-test
    pre_trend_cols = [f"treated_x_t{t}" for t in pre_times]
    r_matrix = np.zeros((len(pre_trend_cols), len(es_model.params)))
    for i, col in enumerate(pre_trend_cols):
        if col in es_model.params.index:
            r_matrix[i, es_model.params.index.get_loc(col)] = 1

    try:
        f_test = es_model.f_test(r_matrix)
        pre_trend_f = float(f_test.fvalue)
        pre_trend_p = float(f_test.pvalue)
    except Exception:
        pre_trend_f = np.nan
        pre_trend_p = np.nan

    return {
        "f_statistic": pre_trend_f,
        "p_value": pre_trend_p,
        "n_pre_coefs": len(pre_trend_cols),
        "pre_times_tested": pre_times,
    }


def main():
    """Main entry point."""
    print("=" * 70)
    print("Module 7 Robustness: Wild Cluster Bootstrap & Restricted Pre-Period")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    # Load and prepare data
    df_refugee = load_refugee_data()
    df_did = prepare_travel_ban_did_data(df_refugee)

    results = {
        "generated": datetime.now(UTC).isoformat(),
        "analyses": {},
    }

    # ==========================================================================
    # 1. Wild Cluster Bootstrap (full pre-period)
    # ==========================================================================
    print("\n" + "#" * 70)
    print("# ANALYSIS 1: WILD CLUSTER BOOTSTRAP (Full Pre-Period)")
    print("#" * 70)

    wcb_full = wild_cluster_bootstrap(df_did, n_bootstrap=1999, seed=42)
    results["analyses"]["wild_cluster_bootstrap_full"] = wcb_full

    # ==========================================================================
    # 2. Randomization Inference (full pre-period)
    # ==========================================================================
    print("\n" + "#" * 70)
    print("# ANALYSIS 2: RANDOMIZATION INFERENCE (Full Pre-Period)")
    print("#" * 70)

    ri_full = randomization_inference(df_did, n_permutations=1999, seed=42)
    results["analyses"]["randomization_inference_full"] = ri_full

    # ==========================================================================
    # 3. Restricted Pre-Period Analysis
    # ==========================================================================
    print("\n" + "#" * 70)
    print("# ANALYSIS 3: RESTRICTED PRE-PERIOD DiD")
    print("#" * 70)

    restricted_results = restricted_preperiod_analysis(df_did)
    results["analyses"]["restricted_preperiod"] = restricted_results

    # ==========================================================================
    # 4. Pre-trend tests for restricted periods
    # ==========================================================================
    print("\n" + "#" * 70)
    print("# ANALYSIS 4: PRE-TREND TESTS BY RESTRICTION")
    print("#" * 70)

    pretrend_tests = {}
    print("\nJoint Pre-Trend F-tests:")
    print(f"{'Pre-Period':<20} {'F-stat':>10} {'p-value':>10} {'# Coefs':>10}")
    print("-" * 55)

    # Full period
    pt_full = run_joint_pretrend_test(df_did, pre_period_start=None)
    pretrend_tests["full"] = pt_full
    print(
        f"{'2002-2017':<20} {pt_full['f_statistic']:>10.2f} "
        f"{pt_full['p_value']:>10.4f} {pt_full['n_pre_coefs']:>10}"
    )

    # Restricted periods
    for start_year in [2010, 2012, 2013, 2014, 2015]:
        pt = run_joint_pretrend_test(df_did, pre_period_start=start_year)
        pretrend_tests[f"from_{start_year}"] = pt
        label = f"{start_year}-2017"
        if not np.isnan(pt["f_statistic"]):
            print(
                f"{label:<20} {pt['f_statistic']:>10.2f} "
                f"{pt['p_value']:>10.4f} {pt['n_pre_coefs']:>10}"
            )
        else:
            print(f"{label:<20} {'N/A':>10} {'N/A':>10} {pt['n_pre_coefs']:>10}")

    results["analyses"]["pretrend_tests"] = pretrend_tests

    # ==========================================================================
    # 5. Wild Cluster Bootstrap with restricted pre-period (2013-2017)
    # ==========================================================================
    print("\n" + "#" * 70)
    print("# ANALYSIS 5: WILD CLUSTER BOOTSTRAP (Restricted Pre-Period 2013-2017)")
    print("#" * 70)

    wcb_restricted = wild_cluster_bootstrap(
        df_did, n_bootstrap=1999, seed=42, pre_period_start=2013
    )
    results["analyses"]["wild_cluster_bootstrap_restricted_2013"] = wcb_restricted

    # ==========================================================================
    # 6. Randomization Inference with restricted pre-period
    # ==========================================================================
    print("\n" + "#" * 70)
    print("# ANALYSIS 6: RANDOMIZATION INFERENCE (Restricted Pre-Period 2013-2017)")
    print("#" * 70)

    ri_restricted = randomization_inference(
        df_did, n_permutations=1999, seed=42, pre_period_start=2013
    )
    results["analyses"]["randomization_inference_restricted_2013"] = ri_restricted

    # ==========================================================================
    # Save results
    # ==========================================================================
    output_path = RESULTS_DIR / "module_7_robustness.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY OF ROBUSTNESS RESULTS")
    print("=" * 70)

    print("\n1. SMALL-SAMPLE INFERENCE (Full Pre-Period 2002-2017):")
    print(f"   Original ATT: {wcb_full['original_att']:.4f}")
    print(f"   Conventional p-value: {wcb_full['conventional_p_value']:.4f}")
    print(f"   Wild Cluster Bootstrap p-value: {wcb_full['bootstrap_p_value']:.4f}")
    print(f"   Randomization Inference p-value: {ri_full['p_value_twosided']:.4f}")

    print("\n2. RESTRICTED PRE-PERIOD (2013-2017):")
    r13 = restricted_results.get("restricted_2013_2017", {})
    print(
        f"   ATT: {r13.get('att', 'N/A'):.4f if isinstance(r13.get('att'), float) else 'N/A'}"
    )
    print(
        f"   Conventional p-value: {r13.get('p_value', 'N/A'):.4f if isinstance(r13.get('p_value'), float) else 'N/A'}"
    )
    print(f"   WCB p-value: {wcb_restricted['bootstrap_p_value']:.4f}")
    print(f"   RI p-value: {ri_restricted['p_value_twosided']:.4f}")

    print("\n3. PRE-TREND TEST IMPROVEMENT:")
    print(
        f"   Full period (2002-2017): F={pt_full['f_statistic']:.2f}, p={pt_full['p_value']:.4f}"
    )
    pt_2013 = pretrend_tests.get("from_2013", {})
    if not np.isnan(pt_2013.get("f_statistic", np.nan)):
        print(
            f"   Restricted (2013-2017): F={pt_2013['f_statistic']:.2f}, "
            f"p={pt_2013['p_value']:.4f}"
        )

    return results


if __name__ == "__main__":
    main()
