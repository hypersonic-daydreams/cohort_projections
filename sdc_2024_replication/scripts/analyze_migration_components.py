#!/usr/bin/env python3
"""
Analyze migration components from Census Bureau Population Estimates.

This script extracts and analyzes the relationship between North Dakota's
international migration and national trends to develop an empirically-grounded
adjustment factor for immigration policy scenarios.

Data sources:
- NST-EST2024-ALLDATA.csv (2020-2024)
- NST-EST2020-ALLDATA.csv (2010-2020)
"""
# mypy: disable-error-code="arg-type,union-attr,assignment,return-value,operator"

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Paths
BASE_DIR = Path(__file__).parent.parent
SOURCE_DIR = BASE_DIR / "data_immigration_policy" / "source"
OUTPUT_DIR = BASE_DIR / "data_immigration_policy" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_2020_2024_data() -> pd.DataFrame:
    """Load 2020-2024 vintage estimates."""
    df = pd.read_csv(SOURCE_DIR / "NST-EST2024-ALLDATA.csv")

    # Filter to state level (SUMLEV=040) plus US total (SUMLEV=010)
    df = df[df["SUMLEV"].isin([10, 40])].copy()

    # Reshape to long format for years 2020-2024
    years = [2020, 2021, 2022, 2023, 2024]
    records = []

    for _, row in df.iterrows():
        for year in years:
            records.append(
                {
                    "state": row["NAME"],
                    "state_fips": row["STATE"],
                    "year": year,
                    "population": row[f"POPESTIMATE{year}"],
                    "pop_change": row[f"NPOPCHG_{year}"],
                    "births": row[f"BIRTHS{year}"],
                    "deaths": row[f"DEATHS{year}"],
                    "natural_change": row[f"NATURALCHG{year}"],
                    "intl_migration": row[f"INTERNATIONALMIG{year}"],
                    "domestic_migration": row[f"DOMESTICMIG{year}"],
                    "net_migration": row[f"NETMIG{year}"],
                }
            )

    return pd.DataFrame(records)


def load_2010_2020_data() -> pd.DataFrame:
    """Load 2010-2020 vintage estimates."""
    df = pd.read_csv(SOURCE_DIR / "NST-EST2020-ALLDATA.csv")

    # Filter to state level (SUMLEV=040) plus US total (SUMLEV=010)
    df = df[df["SUMLEV"].isin([10, 40])].copy()

    # Reshape to long format for years 2010-2020
    years = list(range(2010, 2021))
    records = []

    for _, row in df.iterrows():
        for year in years:
            records.append(
                {
                    "state": row["NAME"],
                    "state_fips": row["STATE"],
                    "year": year,
                    "population": row[f"POPESTIMATE{year}"],
                    "pop_change": row[f"NPOPCHG_{year}"],
                    "births": row[f"BIRTHS{year}"],
                    "deaths": row[f"DEATHS{year}"],
                    "natural_change": row[f"NATURALINC{year}"],  # Different column name
                    "intl_migration": row[f"INTERNATIONALMIG{year}"],
                    "domestic_migration": row[f"DOMESTICMIG{year}"],
                    "net_migration": row[f"NETMIG{year}"],
                }
            )

    return pd.DataFrame(records)


def combine_data() -> pd.DataFrame:
    """Combine 2010-2020 and 2020-2024 data, handling overlap."""
    df_2010 = load_2010_2020_data()
    df_2020 = load_2020_2024_data()

    # Use 2020-2024 vintage for 2020 onwards (more current methodology)
    df_2010 = df_2010[df_2010["year"] < 2020]

    df = pd.concat([df_2010, df_2020], ignore_index=True)
    df = df.sort_values(["state", "year"]).reset_index(drop=True)

    return df


def analyze_nd_migration(df: pd.DataFrame) -> dict:
    """Analyze North Dakota's migration patterns relative to national trends."""
    # Get ND and US data
    nd = df[df["state"] == "North Dakota"].copy()
    us = df[df["state"] == "United States"].copy()

    # Merge for analysis
    merged = pd.merge(nd, us, on="year", suffixes=("_nd", "_us"))

    # Calculate key metrics
    merged["nd_share_intl"] = merged["intl_migration_nd"] / merged["intl_migration_us"]
    merged["nd_intl_pct_of_change"] = merged["intl_migration_nd"] / merged["pop_change_nd"].replace(
        0, np.nan
    )
    merged["us_intl_pct_of_change"] = merged["intl_migration_us"] / merged["pop_change_us"].replace(
        0, np.nan
    )
    merged["nd_share_pop"] = merged["population_nd"] / merged["population_us"]

    results = {
        "years": merged["year"].tolist(),
        "nd_intl_migration": merged["intl_migration_nd"].tolist(),
        "us_intl_migration": merged["intl_migration_us"].tolist(),
        "nd_share_of_us_intl": merged["nd_share_intl"].tolist(),
        "nd_share_of_us_pop": merged["nd_share_pop"].tolist(),
        "nd_intl_as_pct_of_change": merged["nd_intl_pct_of_change"].tolist(),
    }

    # Summary statistics
    results["summary"] = {
        "mean_nd_share_intl": float(merged["nd_share_intl"].mean()),
        "std_nd_share_intl": float(merged["nd_share_intl"].std()),
        "min_nd_share_intl": float(merged["nd_share_intl"].min()),
        "max_nd_share_intl": float(merged["nd_share_intl"].max()),
        "mean_nd_share_pop": float(merged["nd_share_pop"].mean()),
        "ratio_intl_to_pop_share": float(
            merged["nd_share_intl"].mean() / merged["nd_share_pop"].mean()
        ),
    }

    return results, merged


def run_regression(merged: pd.DataFrame) -> dict:
    """Run regression: ND international migration ~ US international migration."""
    x = merged["intl_migration_us"].values
    y = merged["intl_migration_nd"].values

    # Simple linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Also fit through origin (proportional model)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    proportional_coef = y_mean / x_mean if x_mean != 0 else 0

    # Calculate R-squared for proportional model
    y_pred_prop = proportional_coef * x
    ss_res_prop = np.sum((y - y_pred_prop) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2_proportional = 1 - (ss_res_prop / ss_tot) if ss_tot != 0 else 0

    results = {
        "linear_model": {
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "std_err": float(std_err),
            "interpretation": f"For every 1 person increase in US intl migration, ND gets {slope:.6f} people",
        },
        "proportional_model": {
            "coefficient": float(proportional_coef),
            "r_squared": float(r2_proportional),
            "interpretation": f"ND receives {proportional_coef*100:.4f}% of US international migration",
        },
        "recommendation": {
            "model": "proportional" if r2_proportional > r_value**2 * 0.9 else "linear",
            "transfer_coefficient": float(proportional_coef)
            if r2_proportional > r_value**2 * 0.9
            else float(slope),
        },
    }

    return results


def calculate_policy_adjustment(regression: dict, merged: pd.DataFrame) -> dict:
    """Calculate adjustment factors based on CBO immigration policy projections."""
    # CBO projections (from the research report)
    cbo_projections = {
        "2025_original": 1_100_000,  # Original CBO projection
        "2025_revised": -290_000,  # September 2025 revision
        "change": -1_390_000,  # Net change
        "percent_change": -127,  # Percent change
    }

    # Get recent baseline (average of 2022-2024 when international migration was high)
    recent = merged[merged["year"].isin([2022, 2023, 2024])]
    baseline_us_intl = recent["intl_migration_us"].mean()
    baseline_nd_intl = recent["intl_migration_nd"].mean()

    # Transfer coefficient
    transfer_coef = regression["proportional_model"]["coefficient"]

    # Scenario calculations
    scenarios = {}

    # Scenario 1: Full CBO reduction applied
    scenarios["cbo_full"] = {
        "name": "Full CBO Reduction",
        "us_intl_migration": cbo_projections["2025_revised"],
        "nd_intl_migration": cbo_projections["2025_revised"] * transfer_coef,
        "nd_change_from_baseline": (cbo_projections["2025_revised"] * transfer_coef)
        - baseline_nd_intl,
        "adjustment_multiplier": cbo_projections["2025_revised"] / baseline_us_intl
        if baseline_us_intl
        else 0,
    }

    # Scenario 2: 50% of CBO reduction (moderate)
    moderate_us = baseline_us_intl * 0.5
    scenarios["moderate"] = {
        "name": "50% Reduction from Baseline",
        "us_intl_migration": moderate_us,
        "nd_intl_migration": moderate_us * transfer_coef,
        "nd_change_from_baseline": (moderate_us * transfer_coef) - baseline_nd_intl,
        "adjustment_multiplier": 0.5,
    }

    # Scenario 3: Zero international migration
    scenarios["zero_intl"] = {
        "name": "Zero International Migration",
        "us_intl_migration": 0,
        "nd_intl_migration": 0,
        "nd_change_from_baseline": -baseline_nd_intl,
        "adjustment_multiplier": 0.0,
    }

    return {
        "cbo_projections": cbo_projections,
        "baseline": {
            "period": "2022-2024 average",
            "us_intl_migration": float(baseline_us_intl),
            "nd_intl_migration": float(baseline_nd_intl),
        },
        "transfer_coefficient": float(transfer_coef),
        "scenarios": scenarios,
    }


def main():
    """Main analysis function."""
    print("Loading Census Bureau population estimates data...")
    df = combine_data()

    print(f"  Total records: {len(df)}")
    print(f"  States: {df['state'].nunique()}")
    print(f"  Years: {df['year'].min()} to {df['year'].max()}")

    # Save combined data
    df.to_csv(OUTPUT_DIR / "combined_components_of_change.csv", index=False)
    print("\n  Saved: combined_components_of_change.csv")

    # Analyze ND migration
    print("\nAnalyzing North Dakota migration patterns...")
    analysis, merged = analyze_nd_migration(df)

    # Run regression
    print("Running regression analysis...")
    regression = run_regression(merged)

    # Calculate policy adjustments
    print("Calculating policy scenario adjustments...")
    policy = calculate_policy_adjustment(regression, merged)

    # Combine results
    results = {
        "generated": datetime.now(UTC).isoformat(),
        "data_years": f"{df['year'].min()}-{df['year'].max()}",
        "analysis": analysis,
        "regression": regression,
        "policy_scenarios": policy,
    }

    # Save results
    with open(OUTPUT_DIR / "migration_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved: migration_analysis_results.json")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: North Dakota International Migration Analysis")
    print("=" * 60)

    print(f"\nData Period: {df['year'].min()}-{df['year'].max()} ({df['year'].nunique()} years)")

    print("\nND Share of US International Migration:")
    print(f"  Mean:  {analysis['summary']['mean_nd_share_intl']*100:.4f}%")
    print(f"  Std:   {analysis['summary']['std_nd_share_intl']*100:.4f}%")
    print(
        f"  Range: {analysis['summary']['min_nd_share_intl']*100:.4f}% - {analysis['summary']['max_nd_share_intl']*100:.4f}%"
    )

    print(f"\nND Share of US Population: {analysis['summary']['mean_nd_share_pop']*100:.4f}%")
    print(f"Ratio (intl share / pop share): {analysis['summary']['ratio_intl_to_pop_share']:.2f}x")

    print("\nRegression Results:")
    print(f"  Linear Model R²: {regression['linear_model']['r_squared']:.4f}")
    print(f"  Proportional Model R²: {regression['proportional_model']['r_squared']:.4f}")
    print(f"  Transfer Coefficient: {regression['proportional_model']['coefficient']*100:.4f}%")

    print("\nPolicy Scenario Implications:")
    print(
        f"  Baseline ND Intl Migration (2022-24 avg): {policy['baseline']['nd_intl_migration']:,.0f}"
    )
    for _name, scenario in policy["scenarios"].items():
        print(
            f"  {scenario['name']}: {scenario['nd_intl_migration']:,.0f} ({scenario['adjustment_multiplier']:.2%} of baseline)"
        )

    # Create summary CSV
    summary_df = pd.DataFrame(
        {
            "year": analysis["years"],
            "nd_intl_migration": analysis["nd_intl_migration"],
            "us_intl_migration": analysis["us_intl_migration"],
            "nd_share_of_us_intl_pct": [x * 100 for x in analysis["nd_share_of_us_intl"]],
            "nd_share_of_us_pop_pct": [x * 100 for x in analysis["nd_share_of_us_pop"]],
        }
    )
    summary_df.to_csv(OUTPUT_DIR / "nd_migration_summary.csv", index=False)
    print("\n  Saved: nd_migration_summary.csv")

    return results


if __name__ == "__main__":
    main()
