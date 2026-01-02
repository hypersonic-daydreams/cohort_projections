#!/usr/bin/env python3
"""
Prepare immigration policy scenario data for SDC 2024 replication.

This script creates a third variant of the SDC methodology that incorporates
the 2025 immigration policy changes based on empirical analysis of ND's
relationship to national international migration trends.

Methodology:
1. Start with the "updated" data variant (2024 base population, 2023 CDC survival)
2. Calculate the international migration component of SDC's total migration
3. Apply CBO-derived reduction factors to the international component
4. Create adjusted migration rates for each projection period

Key assumptions:
- SDC's migration rates combine domestic and international migration
- We use the empirical ratio of international:total migration to decompose
- The CBO reduction applies only to the international component
- Domestic migration patterns are assumed unchanged

Author: Immigration Policy Scenario Analysis
Date: 2025-12-28
"""
# mypy: disable-error-code="arg-type,union-attr"

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Paths - Use project-level data directories
BASE_DIR = Path(__file__).parent.parent  # sdc_2024_replication/
PROJECT_ROOT = BASE_DIR.parent  # cohort_projections/

# SDC replication data directories
UPDATED_DATA_DIR = BASE_DIR / "data_updated"

# Immigration policy data (now in project-level directories)
ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "rates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_analysis_results() -> dict:
    """Load the migration analysis results."""
    with open(ANALYSIS_DIR / "migration_analysis_results.json") as f:
        return json.load(f)


def calculate_intl_migration_share() -> dict:
    """
    Calculate the share of ND migration that is international vs domestic.

    From the Census components of change analysis:
    - We know ND's international migration (annual)
    - We know ND's domestic migration (annual)
    - We can compute the ratio for recent years
    """
    # Load the combined components data
    df = pd.read_csv(ANALYSIS_DIR / "combined_components_of_change.csv")
    nd = df[df["state"] == "North Dakota"].copy()

    # Calculate shares for each year
    nd["total_migration"] = nd["intl_migration"] + nd["domestic_migration"]
    nd["intl_share"] = nd["intl_migration"] / nd["total_migration"].replace(0, float("nan"))

    # Use different periods for different scenarios
    # Pre-Bakken (2010-2013): More typical pattern
    # Bakken boom (2014-2017): High domestic in-migration
    # Post-Bakken (2018-2024): Transition period

    pre_bakken = nd[nd["year"].between(2010, 2013)]["intl_share"].mean()
    bakken = nd[nd["year"].between(2014, 2017)]["intl_share"].mean()
    post_bakken = nd[nd["year"].between(2018, 2024)]["intl_share"].mean()
    recent = nd[nd["year"].between(2022, 2024)]["intl_share"].mean()
    overall = nd["intl_share"].mean()

    # For SDC's 2000-2020 based rates, use overall average
    # But weight toward recent years since that's the trend

    return {
        "pre_bakken_2010_2013": float(pre_bakken),
        "bakken_2014_2017": float(bakken),
        "post_bakken_2018_2024": float(post_bakken),
        "recent_2022_2024": float(recent),
        "overall_2010_2024": float(overall),
        "recommended_for_sdc": float(overall),  # Use overall for SDC rates
        "by_year": nd[
            ["year", "intl_migration", "domestic_migration", "total_migration", "intl_share"]
        ].to_dict("records"),
    }


def calculate_policy_adjustment_factors(analysis: dict, intl_share: dict) -> dict:
    """
    Calculate adjustment factors for migration rates under different policy scenarios.

    The logic:
    1. SDC migration rate = domestic_rate + international_rate
    2. Under policy change, international_rate is reduced
    3. New rate = domestic_rate + (international_rate × policy_multiplier)
    4. New rate = SDC_rate - international_rate × (1 - policy_multiplier)
    5. New rate = SDC_rate × [1 - intl_share × (1 - policy_multiplier)]

    So the adjustment multiplier for SDC rates is:
    adjustment = 1 - intl_share × (1 - policy_multiplier)
    """
    cbo = analysis["policy_scenarios"]["cbo_projections"]
    baseline_us = analysis["policy_scenarios"]["baseline"]["us_intl_migration"]

    # Policy multipliers (what fraction of baseline intl migration remains)
    policy_multipliers = {
        "full_cbo": cbo["2025_revised"] / baseline_us,  # Negative (net outflow)
        "moderate": 0.5,  # 50% of baseline
        "conservative": 0.7,  # 30% reduction
        "zero_intl": 0.0,  # Complete cessation
    }

    # Calculate rate adjustments for each scenario
    # Using the overall international share from components analysis
    intl_proportion = intl_share["recommended_for_sdc"]

    adjustments = {}
    for scenario, policy_mult in policy_multipliers.items():
        # Adjustment = 1 - intl_share × (1 - policy_multiplier)
        rate_adjustment = 1 - intl_proportion * (1 - policy_mult)
        adjustments[scenario] = {
            "policy_multiplier": float(policy_mult),
            "rate_adjustment": float(rate_adjustment),
            "interpretation": f"Multiply SDC migration rates by {rate_adjustment:.4f}",
        }

    return {
        "intl_share_of_migration": float(intl_proportion),
        "scenarios": adjustments,
    }


def create_adjusted_migration_rates(adjustment_factor: float) -> pd.DataFrame:
    """Create adjusted migration rates for the policy scenario."""
    # Load original updated data migration rates
    original = pd.read_csv(UPDATED_DATA_DIR / "migration_rates_by_county.csv")

    # Apply adjustment factor
    adjusted = original.copy()
    adjusted["migration_rate"] = original["migration_rate"] * adjustment_factor

    return adjusted


def create_period_specific_multipliers(base_multipliers: dict, adjustment: float) -> dict:
    """
    Create period-specific multipliers that incorporate the policy adjustment.

    The SDC uses period multipliers: 0.2, 0.6, 0.6, 0.5, 0.7, 0.7
    We need to further adjust these for the policy impact.

    Options:
    1. Apply same adjustment to all periods (policy persists)
    2. Phase in adjustment (gradual policy impact)
    3. Phase out adjustment (policy normalizes)

    We'll implement option 1 (persistent) as the main scenario,
    but document alternatives.
    """
    # Original SDC period multipliers
    sdc_multipliers = {
        2025: 0.2,
        2030: 0.6,
        2035: 0.6,
        2040: 0.5,
        2045: 0.7,
        2050: 0.7,
    }

    # Apply policy adjustment to the migration component
    # The period multiplier already dampens migration
    # We further reduce by the policy factor
    adjusted_multipliers = {}
    for year, mult in sdc_multipliers.items():
        # The adjustment factor already accounts for intl share
        # So we multiply the period multiplier by it
        adjusted_multipliers[year] = mult * adjustment

    return {
        "original_sdc": sdc_multipliers,
        "policy_adjusted": adjusted_multipliers,
        "adjustment_factor": adjustment,
    }


def copy_unchanged_data():
    """Copy data that doesn't change from the updated variant."""
    files_to_copy = [
        "base_population_by_county.csv",
        "fertility_rates_by_county.csv",
        "survival_rates_by_county.csv",
        "adjustment_factors_by_county.csv",
    ]

    for filename in files_to_copy:
        src = UPDATED_DATA_DIR / filename
        dst = OUTPUT_DIR / filename
        if src.exists():
            shutil.copy(src, dst)
            print(f"  Copied: {filename}")


def write_manifest(intl_share: dict, adjustments: dict, scenario: str):
    """Write manifest documenting the data sources and adjustments."""
    adj = adjustments["scenarios"][scenario]

    manifest = f"""# Immigration Policy Scenario Data Manifest

**Generated:** {datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")}
**Scenario:** {scenario.replace("_", " ").title()}

---

## Data Sources

### Base Population
- **Source:** Census Vintage 2024 (same as "updated" variant)
- **Description:** 2020 Census age/sex distribution scaled to 2024 totals

### Survival Rates
- **Source:** CDC National Life Tables 2023 (same as "updated" variant)
- **Description:** 5-year survival probabilities from 2023 national life tables

### Fertility Rates
- **Source:** SDC 2024 Original (same as "updated" variant)
- **Description:** Original SDC county-level fertility patterns

### Migration Rates
- **Source:** SDC 2024 Original × Policy Adjustment Factor
- **Description:** SDC migration rates adjusted for 2025 immigration policy changes

---

## Policy Adjustment Methodology

### Empirical Basis
- **Data:** Census Bureau Components of Population Change (2010-2024)
- **Analysis:** Regression of ND international migration on US international migration
- **Transfer Coefficient:** ND receives {adjustments['intl_share_of_migration']*100:.2f}% of its migration from international sources

### Adjustment Calculation
1. SDC migration rates combine domestic + international migration
2. International share of ND migration: {intl_share['recommended_for_sdc']*100:.2f}%
3. Policy multiplier for "{scenario}": {adj['policy_multiplier']:.4f}
4. Rate adjustment factor: {adj['rate_adjustment']:.4f}

**Formula:** `new_rate = sdc_rate × {adj['rate_adjustment']:.4f}`

### Interpretation
{adj['interpretation']}

---

## Scenario Details

### {scenario.replace("_", " ").title()} Scenario

Based on CBO September 2025 demographic outlook revision:
- Original 2025 projection: +1.1 million net international migration
- Revised 2025 projection: -290,000 net international migration
- Policy multiplier applied: {adj['policy_multiplier']:.2%} of baseline

---

## Files in This Directory

1. `base_population_by_county.csv` - Same as updated variant
2. `fertility_rates_by_county.csv` - Same as updated variant
3. `survival_rates_by_county.csv` - Same as updated variant
4. `adjustment_factors_by_county.csv` - Same as updated variant
5. `migration_rates_by_county.csv` - **ADJUSTED** for policy scenario
6. `period_multipliers.json` - Period-specific migration adjustments

---

## References

- [ADR-018: Immigration Policy Scenario Methodology](../../docs/governance/adrs/018-immigration-policy-scenario-methodology.md)
- [CBO Demographic Outlook Update 2025-2055](https://www.cbo.gov/publication/61735)
- [Immigration Policy Research Report](../../docs/research/2025_immigration_policy_demographic_impact.md)
"""

    with open(OUTPUT_DIR / "MANIFEST.md", "w") as f:
        f.write(manifest)


def main():
    """Main function to prepare immigration policy scenario data."""
    print("=" * 60)
    print("Preparing Immigration Policy Scenario Data")
    print("=" * 60)

    # Load analysis results
    print("\nLoading migration analysis results...")
    analysis = load_analysis_results()

    # Calculate international migration share
    print("Calculating international migration share...")
    intl_share = calculate_intl_migration_share()
    print(f"  International share of ND migration: {intl_share['recommended_for_sdc']*100:.2f}%")
    print(f"  Recent (2022-24): {intl_share['recent_2022_2024']*100:.2f}%")

    # Calculate adjustment factors
    print("\nCalculating policy adjustment factors...")
    adjustments = calculate_policy_adjustment_factors(analysis, intl_share)

    print("\nScenario adjustment factors:")
    for name, adj in adjustments["scenarios"].items():
        print(f"  {name}: {adj['rate_adjustment']:.4f} ({adj['interpretation']})")

    # Use the "full_cbo" scenario as the primary policy variant
    scenario = "full_cbo"
    adjustment_factor = adjustments["scenarios"][scenario]["rate_adjustment"]

    print(f"\nUsing '{scenario}' scenario (adjustment factor: {adjustment_factor:.4f})")

    # Create adjusted migration rates
    print("\nCreating adjusted migration rates...")
    adjusted_rates = create_adjusted_migration_rates(adjustment_factor)
    adjusted_rates.to_csv(OUTPUT_DIR / "migration_rates_by_county.csv", index=False)
    print("  Saved: migration_rates_by_county.csv")

    # Create period multipliers
    print("Creating period-specific multipliers...")
    period_mult = create_period_specific_multipliers({}, adjustment_factor)
    with open(OUTPUT_DIR / "period_multipliers.json", "w") as f:
        json.dump(period_mult, f, indent=2)
    print("  Saved: period_multipliers.json")

    # Copy unchanged data
    print("\nCopying unchanged data files...")
    copy_unchanged_data()

    # Write manifest
    print("\nWriting manifest...")
    write_manifest(intl_share, adjustments, scenario)
    print("  Saved: MANIFEST.md")

    # Save full adjustment details
    full_results = {
        "generated": datetime.now(UTC).isoformat(),
        "scenario": scenario,
        "intl_share_analysis": intl_share,
        "adjustment_factors": adjustments,
        "period_multipliers": period_mult,
    }
    with open(OUTPUT_DIR / "adjustment_details.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print("  Saved: adjustment_details.json")

    print("\n" + "=" * 60)
    print("Immigration Policy Scenario Data Preparation Complete")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    print(f"  Scenario: {scenario}")
    print(f"  International share: {intl_share['recommended_for_sdc']*100:.2f}%")
    print(f"  Rate adjustment: {adjustment_factor:.4f}")
    print(f"  Output directory: {OUTPUT_DIR}")

    return full_results


if __name__ == "__main__":
    main()
