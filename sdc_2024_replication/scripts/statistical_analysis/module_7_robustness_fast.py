#!/usr/bin/env python3
"""
Module 7 Robustness (Fast Version): Wild Cluster Bootstrap and Restricted Pre-Period DiD
=========================================================================================

Implements causal inference robustness checks for P3.10 and P3.14:
1. Wild cluster bootstrap for DiD ATT (optimized for speed)
2. Randomization/permutation inference
3. Restricted pre-period DiD (using years with more parallel trends)

Usage:
    python module_7_robustness_fast.py
    python module_7_robustness_fast.py --rigorous
    python module_7_robustness_fast.py --n-bootstrap 9999 --n-permutations 9999
"""

import argparse
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

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True)


def load_refugee_data() -> pd.DataFrame:
    """Load refugee arrivals data."""
    refugee_path = DATA_DIR / "refugee_arrivals_by_state_nationality.parquet"
    df_refugee = pd.read_parquet(refugee_path)
    print(f"Loaded refugee arrivals: {df_refugee.shape}")
    return df_refugee


def prepare_travel_ban_did_data(df_refugee: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for Travel Ban DiD analysis."""
    travel_ban_countries = [
        "Iran",
        "Iraq",
        "Libya",
        "Somalia",
        "Sudan",
        "Syria",
        "Yemen",
    ]

    df_nat_year = (
        df_refugee.groupby(["fiscal_year", "nationality"])["arrivals"]
        .sum()
        .reset_index()
    )
    df_nat_year.columns = ["year", "nationality", "arrivals"]

    # Exclude aggregate pseudo-nationalities that are not country units.
    pseudo_nationalities = {"total", "fy refugee admissions"}
    nat_lower = (
        df_nat_year["nationality"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    df_nat_year = df_nat_year.loc[~nat_lower.isin(pseudo_nationalities)].copy()

    df_nat_year["treated"] = (
        df_nat_year["nationality"].isin(travel_ban_countries).astype(int)
    )
    df_nat_year["post"] = (df_nat_year["year"] >= 2018).astype(int)
    df_nat_year["treated_x_post"] = df_nat_year["treated"] * df_nat_year["post"]
    df_nat_year["log_arrivals"] = np.log(df_nat_year["arrivals"] + 1)
    df_nat_year["rel_time"] = df_nat_year["year"] - 2018

    return df_nat_year


def estimate_did_simple(df: pd.DataFrame, pre_period_start: int | None = None) -> dict:
    """
    Estimate TWFE DiD model with simple OLS (faster than clustered SE for bootstrap).
    """
    df_analysis = df[df["year"] < 2020].copy()
    if pre_period_start is not None:
        df_analysis = df_analysis[
            (df_analysis["year"] >= pre_period_start) | (df_analysis["post"] == 1)
        ].copy()

    df_analysis = df_analysis.reset_index(drop=True)

    y = df_analysis["log_arrivals"].values
    X = pd.DataFrame({"treated_x_post": df_analysis["treated_x_post"].values})

    nationality_dummies = pd.get_dummies(
        df_analysis["nationality"], prefix="nat", drop_first=True
    )
    year_dummies = pd.get_dummies(df_analysis["year"], prefix="year", drop_first=True)

    X = pd.concat([X, nationality_dummies.reset_index(drop=True)], axis=1)
    X = pd.concat([X, year_dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X).astype(float)

    model = OLS(y, X).fit()
    att = model.params["treated_x_post"]

    # Clustered SE version for comparison
    model_cluster = OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": df_analysis["nationality"]}
    )

    return {
        "att": float(att),
        "se_robust": float(model.bse["treated_x_post"]),
        "se_clustered": float(model_cluster.bse["treated_x_post"]),
        "t_stat": float(model_cluster.tvalues["treated_x_post"]),
        "p_value": float(model_cluster.pvalues["treated_x_post"]),
        "ci_lower": float(model_cluster.conf_int().loc["treated_x_post", 0]),
        "ci_upper": float(model_cluster.conf_int().loc["treated_x_post", 1]),
        "df_analysis": df_analysis,
        "y": y,
        "X": X,
    }


def wild_cluster_bootstrap_fast(
    df: pd.DataFrame,
    n_bootstrap: int = 999,
    seed: int = 42,
    pre_period_start: int | None = None,
) -> dict:
    """
    Fast wild cluster bootstrap using precomputed matrices.
    """
    print("\n" + "=" * 60)
    print("WILD CLUSTER BOOTSTRAP FOR DiD ATT")
    print("=" * 60)

    np.random.seed(seed)

    base = estimate_did_simple(df, pre_period_start)
    df_analysis = base["df_analysis"]
    att_original = base["att"]

    print(f"\nOriginal ATT: {att_original:.4f}")
    print(f"Clustered SE: {base['se_clustered']:.4f}")
    print(f"Conventional p-value: {base['p_value']:.4f}")

    clusters = df_analysis["nationality"].unique()
    n_clusters = len(clusters)
    n_treated = df_analysis[df_analysis["treated"] == 1]["nationality"].nunique()

    print(f"\nClusters: {n_clusters} total, {n_treated} treated")
    print(f"Bootstrap iterations: {n_bootstrap}")

    # Precompute design matrices
    y = df_analysis["log_arrivals"].values.copy()

    # H0 model (no treatment effect)
    X_h0 = pd.DataFrame()
    nationality_dummies = pd.get_dummies(
        df_analysis["nationality"], prefix="nat", drop_first=True
    )
    year_dummies = pd.get_dummies(df_analysis["year"], prefix="year", drop_first=True)
    X_h0 = pd.concat([X_h0, nationality_dummies.reset_index(drop=True)], axis=1)
    X_h0 = pd.concat([X_h0, year_dummies.reset_index(drop=True)], axis=1)
    X_h0 = sm.add_constant(X_h0).astype(float)

    model_h0 = OLS(y, X_h0).fit()
    residuals_h0 = model_h0.resid
    fitted_h0 = model_h0.fittedvalues

    # Full model design
    X_full = pd.DataFrame({"treated_x_post": df_analysis["treated_x_post"].values})
    X_full = pd.concat([X_full, nationality_dummies.reset_index(drop=True)], axis=1)
    X_full = pd.concat([X_full, year_dummies.reset_index(drop=True)], axis=1)
    X_full = sm.add_constant(X_full).astype(float)
    X_full_arr = X_full.values

    # Precompute (X'X)^{-1}X' for fast coefficient extraction
    XtX_inv = np.linalg.inv(X_full_arr.T @ X_full_arr)
    XtX_inv_Xt = XtX_inv @ X_full_arr.T

    # Precompute cluster masks
    cluster_masks = {c: (df_analysis["nationality"] == c).values for c in clusters}

    # Bootstrap loop
    bootstrap_atts = []
    att_idx = X_full.columns.get_loc("treated_x_post")

    for _ in range(n_bootstrap):
        # Rademacher weights at cluster level
        weights = np.random.choice([-1.0, 1.0], size=n_clusters)

        # Apply weights to residuals
        weighted_residuals = np.zeros_like(residuals_h0)
        for c_idx, c_name in enumerate(clusters):
            weighted_residuals[cluster_masks[c_name]] = (
                residuals_h0[cluster_masks[c_name]] * weights[c_idx]
            )

        # Bootstrap outcome
        y_boot = fitted_h0 + weighted_residuals

        # Fast coefficient extraction (no full OLS solve)
        beta_boot = XtX_inv_Xt @ y_boot
        bootstrap_atts.append(beta_boot[att_idx])

    bootstrap_atts = np.array(bootstrap_atts)

    # Bootstrap p-value (two-sided)
    p_bootstrap = np.mean(np.abs(bootstrap_atts) >= np.abs(att_original))

    # Bootstrap SE
    bootstrap_se = np.std(bootstrap_atts)

    # Percentile CI
    ci_lower_boot = np.percentile(bootstrap_atts, 2.5)
    ci_upper_boot = np.percentile(bootstrap_atts, 97.5)

    print(f"\n{'='*60}")
    print("WILD CLUSTER BOOTSTRAP RESULTS")
    print("=" * 60)
    print(f"\nConventional p-value: {base['p_value']:.4f}")
    print(f"Bootstrap p-value: {p_bootstrap:.4f}")
    print(f"\nConventional 95% CI: [{base['ci_lower']:.4f}, {base['ci_upper']:.4f}]")
    print(f"Bootstrap 95% CI: [{ci_lower_boot:.4f}, {ci_upper_boot:.4f}]")
    print(f"\nCluster SE: {base['se_clustered']:.4f}")
    print(f"Bootstrap SE: {bootstrap_se:.4f}")

    return {
        "method": "Wild Cluster Bootstrap (Rademacher weights)",
        "n_clusters": n_clusters,
        "n_treated_clusters": n_treated,
        "n_bootstrap": n_bootstrap,
        "original_att": att_original,
        "original_se_clustered": base["se_clustered"],
        "original_t_stat": base["t_stat"],
        "conventional_p_value": base["p_value"],
        "bootstrap_p_value": p_bootstrap,
        "bootstrap_se": float(bootstrap_se),
        "conventional_ci_lower": base["ci_lower"],
        "conventional_ci_upper": base["ci_upper"],
        "bootstrap_ci_lower": float(ci_lower_boot),
        "bootstrap_ci_upper": float(ci_upper_boot),
    }


def randomization_inference_fast(
    df: pd.DataFrame,
    n_permutations: int = 999,
    seed: int = 42,
    pre_period_start: int | None = None,
) -> dict:
    """
    Fast randomization inference using precomputed matrices.
    """
    print("\n" + "=" * 60)
    print("RANDOMIZATION INFERENCE FOR DiD ATT")
    print("=" * 60)

    np.random.seed(seed)

    base = estimate_did_simple(df, pre_period_start)
    df_analysis = base["df_analysis"].copy()
    att_original = base["att"]

    print(f"\nOriginal ATT: {att_original:.4f}")
    print(f"Permutations: {n_permutations}")

    # Get cluster-level treatment
    cluster_treatment = (
        df_analysis.groupby("nationality")["treated"].first().reset_index()
    )
    nationalities = cluster_treatment["nationality"].values
    original_treatment = cluster_treatment["treated"].values.astype(float)
    n_treated = int(original_treatment.sum())

    print(f"Treated clusters: {n_treated} of {len(nationalities)}")

    # Precompute fixed effects design
    y = df_analysis["log_arrivals"].values
    nationality_dummies = pd.get_dummies(
        df_analysis["nationality"], prefix="nat", drop_first=True
    )
    year_dummies = pd.get_dummies(df_analysis["year"], prefix="year", drop_first=True)

    # Build full design matrix template (we'll update treated_x_post column)
    X_template = pd.DataFrame({"treated_x_post": np.zeros(len(df_analysis))})
    X_template = pd.concat(
        [X_template, nationality_dummies.reset_index(drop=True)], axis=1
    )
    X_template = pd.concat([X_template, year_dummies.reset_index(drop=True)], axis=1)
    X_template = sm.add_constant(X_template).astype(float)
    X_arr = X_template.values.copy()

    # Get column index for treated_x_post
    att_idx = X_template.columns.get_loc("treated_x_post")
    post_mask = (df_analysis["post"] == 1).values

    # Permutation loop
    permutation_atts = []

    for _ in range(n_permutations):
        # Permute treatment at cluster level
        perm_treatment = np.random.permutation(original_treatment)
        perm_map = dict(zip(nationalities, perm_treatment, strict=False))

        # Update treated_x_post column
        treated_perm = df_analysis["nationality"].map(perm_map).values
        X_arr[:, att_idx] = treated_perm * post_mask

        # Fast coefficient extraction
        try:
            XtX_inv = np.linalg.inv(X_arr.T @ X_arr)
            beta = XtX_inv @ X_arr.T @ y
            permutation_atts.append(beta[att_idx])
        except np.linalg.LinAlgError:
            continue

    permutation_atts = np.array(permutation_atts)

    # p-values
    p_twosided = np.mean(np.abs(permutation_atts) >= np.abs(att_original))
    p_onesided = np.mean(permutation_atts <= att_original)

    rank = np.sum(permutation_atts <= att_original) + 1

    print(f"\n{'='*60}")
    print("RANDOMIZATION INFERENCE RESULTS")
    print("=" * 60)
    print(f"Valid permutations: {len(permutation_atts)}")
    print(f"\nOriginal ATT: {att_original:.4f}")
    print(
        f"Permutation ATTs: mean={np.mean(permutation_atts):.4f}, SD={np.std(permutation_atts):.4f}"
    )
    print(f"\nRank of original: {rank} / {len(permutation_atts) + 1}")
    print(f"Two-sided p-value: {p_twosided:.4f}")
    print(f"One-sided p-value (negative): {p_onesided:.4f}")

    return {
        "method": "Randomization Inference (Fisher exact test)",
        "n_permutations": n_permutations,
        "n_valid": len(permutation_atts),
        "n_treated_clusters": n_treated,
        "n_total_clusters": len(nationalities),
        "original_att": att_original,
        "permutation_att_mean": float(np.mean(permutation_atts)),
        "permutation_att_sd": float(np.std(permutation_atts)),
        "rank": int(rank),
        "p_value_twosided": p_twosided,
        "p_value_onesided": p_onesided,
    }


def restricted_preperiod_analysis(df: pd.DataFrame) -> dict:
    """Estimate DiD with restricted pre-periods."""
    print("\n" + "=" * 60)
    print("RESTRICTED PRE-PERIOD DiD ANALYSIS")
    print("=" * 60)

    results = {}

    restrictions = [
        (None, "2002-2017 (full)"),
        (2010, "2010-2017 (8 years)"),
        (2012, "2012-2017 (6 years)"),
        (2013, "2013-2017 (5 years)"),
        (2014, "2014-2017 (4 years)"),
        (2015, "2015-2017 (3 years)"),
    ]

    print(f"\n{'Pre-Period':<20} {'ATT':>10} {'SE':>10} {'p-value':>10} {'Sig':>6}")
    print("-" * 60)

    for start_year, label in restrictions:
        res = estimate_did_simple(df, pre_period_start=start_year)
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
            f"{label:<20} {res['att']:>10.4f} {res['se_clustered']:>10.4f} "
            f"{res['p_value']:>10.4f} {sig:>6}"
        )

        key = f"from_{start_year}" if start_year else "full"
        results[key] = {
            "pre_period": label,
            "att": res["att"],
            "se": res["se_clustered"],
            "t_stat": res["t_stat"],
            "p_value": res["p_value"],
            "ci_lower": res["ci_lower"],
            "ci_upper": res["ci_upper"],
        }

    return results


def run_joint_pretrend_test(
    df: pd.DataFrame, pre_period_start: int | None = None
) -> dict:
    """Run joint F-test for pre-trends."""
    df_analysis = df[df["year"] < 2020].copy()
    if pre_period_start is not None:
        df_analysis = df_analysis[
            (df_analysis["year"] >= pre_period_start) | (df_analysis["post"] == 1)
        ].copy()

    df_analysis = df_analysis.reset_index(drop=True)
    df_analysis["rel_time"] = df_analysis["year"] - 2018

    rel_times = sorted(df_analysis["rel_time"].unique())
    rel_times = [int(t) for t in rel_times]
    rel_times_no_ref = [t for t in rel_times if t != -1]
    pre_times = [t for t in rel_times_no_ref if t < 0]

    if len(pre_times) == 0:
        return {"f_statistic": np.nan, "p_value": np.nan, "n_pre_coefs": 0}

    for t in rel_times_no_ref:
        df_analysis[f"treated_x_t{t}"] = (
            df_analysis["treated"] * (df_analysis["rel_time"] == t)
        ).astype(float)

    y = df_analysis["log_arrivals"].values
    interaction_cols = [f"treated_x_t{t}" for t in rel_times_no_ref]

    X = pd.DataFrame({col: df_analysis[col].values for col in interaction_cols})
    nat_dummies = pd.get_dummies(
        df_analysis["nationality"], prefix="nat", drop_first=True
    )
    year_dummies = pd.get_dummies(df_analysis["year"], prefix="year", drop_first=True)
    X = pd.concat([X, nat_dummies.reset_index(drop=True)], axis=1)
    X = pd.concat([X, year_dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X).astype(float)

    es_model = OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": df_analysis["nationality"]}
    )

    pre_trend_cols = [f"treated_x_t{t}" for t in pre_times]
    r_matrix = np.zeros((len(pre_trend_cols), len(es_model.params)))
    for i, col in enumerate(pre_trend_cols):
        if col in es_model.params.index:
            r_matrix[i, es_model.params.index.get_loc(col)] = 1

    try:
        f_test = es_model.f_test(r_matrix)
        return {
            "f_statistic": float(f_test.fvalue),
            "p_value": float(f_test.pvalue),
            "n_pre_coefs": len(pre_trend_cols),
            "pre_times": pre_times,
        }
    except Exception:
        return {
            "f_statistic": np.nan,
            "p_value": np.nan,
            "n_pre_coefs": len(pre_trend_cols),
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Fast Module 7 robustness checks (Travel Ban DiD): wild cluster bootstrap, "
            "randomization inference, and restricted pre-period diagnostics."
        )
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=999,
        help="Wild cluster bootstrap iterations (default: 999).",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=999,
        help="Randomization inference permutations (default: 999).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap/permutation (default: 42).",
    )
    parser.add_argument(
        "--rigorous",
        action="store_true",
        help="Convenience flag: set both --n-bootstrap and --n-permutations to 9999.",
    )
    args = parser.parse_args()
    if args.rigorous:
        args.n_bootstrap = 9999
        args.n_permutations = 9999

    print("=" * 70)
    print("Module 7 Robustness (Fast): Wild Cluster Bootstrap & Restricted Pre-Period")
    print(f"Started: {datetime.now(UTC).isoformat()}")
    print("=" * 70)

    df_refugee = load_refugee_data()
    df_did = prepare_travel_ban_did_data(df_refugee)

    results = {"generated": datetime.now(UTC).isoformat(), "analyses": {}}

    # 1. Wild Cluster Bootstrap (full pre-period)
    print("\n" + "#" * 70)
    print("# ANALYSIS 1: WILD CLUSTER BOOTSTRAP (Full Pre-Period)")
    print("#" * 70)
    wcb_full = wild_cluster_bootstrap_fast(
        df_did, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    results["analyses"]["wild_cluster_bootstrap_full"] = wcb_full

    # 2. Randomization Inference (full pre-period)
    print("\n" + "#" * 70)
    print("# ANALYSIS 2: RANDOMIZATION INFERENCE (Full Pre-Period)")
    print("#" * 70)
    ri_full = randomization_inference_fast(
        df_did, n_permutations=args.n_permutations, seed=args.seed
    )
    results["analyses"]["randomization_inference_full"] = ri_full

    # 3. Restricted Pre-Period Analysis
    print("\n" + "#" * 70)
    print("# ANALYSIS 3: RESTRICTED PRE-PERIOD DiD")
    print("#" * 70)
    restricted = restricted_preperiod_analysis(df_did)
    results["analyses"]["restricted_preperiod"] = restricted

    # 4. Pre-trend tests
    print("\n" + "#" * 70)
    print("# ANALYSIS 4: PRE-TREND TESTS BY RESTRICTION")
    print("#" * 70)
    pretrend_tests = {}
    print(f"\n{'Pre-Period':<20} {'F-stat':>10} {'p-value':>10} {'# Coefs':>10}")
    print("-" * 55)

    for start_year, label in [
        (None, "2002-2017"),
        (2010, "2010-2017"),
        (2012, "2012-2017"),
        (2013, "2013-2017"),
        (2014, "2014-2017"),
        (2015, "2015-2017"),
    ]:
        pt = run_joint_pretrend_test(df_did, pre_period_start=start_year)
        key = f"from_{start_year}" if start_year else "full"
        pretrend_tests[key] = pt
        if not np.isnan(pt["f_statistic"]):
            print(
                f"{label:<20} {pt['f_statistic']:>10.2f} {pt['p_value']:>10.4f} {pt['n_pre_coefs']:>10}"
            )
        else:
            print(f"{label:<20} {'N/A':>10} {'N/A':>10} {pt['n_pre_coefs']:>10}")

    results["analyses"]["pretrend_tests"] = pretrend_tests

    # 5. WCB with restricted pre-period (2013-2017)
    print("\n" + "#" * 70)
    print("# ANALYSIS 5: WILD CLUSTER BOOTSTRAP (Restricted 2013-2017)")
    print("#" * 70)
    wcb_restricted = wild_cluster_bootstrap_fast(
        df_did, n_bootstrap=args.n_bootstrap, seed=args.seed, pre_period_start=2013
    )
    results["analyses"]["wild_cluster_bootstrap_restricted_2013"] = wcb_restricted

    # 6. RI with restricted pre-period
    print("\n" + "#" * 70)
    print("# ANALYSIS 6: RANDOMIZATION INFERENCE (Restricted 2013-2017)")
    print("#" * 70)
    ri_restricted = randomization_inference_fast(
        df_did,
        n_permutations=args.n_permutations,
        seed=args.seed,
        pre_period_start=2013,
    )
    results["analyses"]["randomization_inference_restricted_2013"] = ri_restricted

    # Save results
    output_path = RESULTS_DIR / "module_7_robustness.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ROBUSTNESS RESULTS")
    print("=" * 70)

    print("\n1. SMALL-SAMPLE INFERENCE (Full Pre-Period 2002-2017):")
    print(f"   Original ATT: {wcb_full['original_att']:.4f}")
    print(f"   Conventional p-value: {wcb_full['conventional_p_value']:.4f}")
    print(f"   Wild Cluster Bootstrap p-value: {wcb_full['bootstrap_p_value']:.4f}")
    print(f"   Randomization Inference p-value: {ri_full['p_value_twosided']:.4f}")

    print("\n2. RESTRICTED PRE-PERIOD (2013-2017):")
    r13 = restricted.get("from_2013", {})
    print(f"   ATT: {r13.get('att', 'N/A'):.4f}")
    print(f"   Conventional p-value: {r13.get('p_value', 'N/A'):.4f}")
    print(f"   WCB p-value: {wcb_restricted['bootstrap_p_value']:.4f}")
    print(f"   RI p-value: {ri_restricted['p_value_twosided']:.4f}")

    print("\n3. PRE-TREND TEST IMPROVEMENT:")
    pt_full = pretrend_tests.get("full", {})
    pt_2013 = pretrend_tests.get("from_2013", {})
    print(
        f"   Full (2002-2017): F={pt_full.get('f_statistic', 'N/A'):.2f}, "
        f"p={pt_full.get('p_value', 'N/A'):.4f}"
    )
    if not np.isnan(pt_2013.get("f_statistic", np.nan)):
        print(
            f"   Restricted (2013-2017): F={pt_2013['f_statistic']:.2f}, "
            f"p={pt_2013['p_value']:.4f}"
        )

    return results


if __name__ == "__main__":
    main()
