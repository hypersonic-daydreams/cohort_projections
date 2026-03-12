#!/usr/bin/env python3
"""
Replicate SDC 2024 projection methodology using latest data (2025 base, 2000-2025 migration).

Created: 2026-03-02
Author: Claude Code (automated)

Purpose
-------
The North Dakota State Data Center (SDC) published population projections in 2024 using a
simplified cohort-component method. This script replicates that exact methodology but
substitutes our most recent data inputs: Census PEP Vintage 2025 base population and
residual migration rates computed from 2000-2024 population estimates. The purpose is to
isolate the effect of updated data from the effect of methodological improvements in our
full projection engine.

Method
------
1. Load 2025 base population from processed parquet, aggregate single-year ages to 5-year
   groups and collapse race/ethnicity (SDC uses total population only).
2. Load CDC ND 2020 survival rates (constant, no mortality improvement).
3. Load SDC 2018-2022 blended fertility rates (annual ASFRs by 5-year age group).
4. Load residual migration rates for all available periods (2000-2005 through 2020-2024),
   convert annualized rates to 5-year rates, compute simple arithmetic average across
   periods, and apply flat 60% Bakken dampening to Williams, McKenzie, Mountrail, Dunn,
   and Stark counties.
5. Project forward in 5-year steps from 2025 to 2055 using the cohort-component algorithm:
   survive -> migrate -> births -> birth survival -> migrate 0-4.
6. State total is the sum of all 53 county projections.
7. Write output CSVs and print comparison against SDC 2024 original and our baseline.

Key design decisions
--------------------
- **Flat Bakken dampening at 60%**: Matches the SDC 2024 approach of uniformly reducing
  migration rates for oil-producing counties by 40%. Our full engine uses ADR-051 analysis
  (which rejected dampening) and convergence schedules instead.
- **Simple period averaging**: All migration periods weighted equally. Our full engine uses
  BEBR-style multiperiod weighting (ADR-036) which emphasizes recent trends.
- **No GQ separation**: Group quarters population remains combined in totals, matching
  SDC practice. Our full engine separates GQ per ADR-055.
- **No convergence schedule**: The same averaged migration rate applies to all future
  periods. Our full engine converges rates toward long-run means over time.
- **Constant mortality and fertility**: No improvement factors applied. SDC uses static
  CDC ND 2020 life tables and 2018-2022 blended ASFRs.
- **Sex ratio at birth**: 51.2% male / 48.8% female, matching SDC assumption.
- **2020-2024 period**: This period is 4 years, not 5. Since migration rates are already
  annualized, the conversion to 5-year rates uses (1+r)^5-1 uniformly for all periods.

Validation results (2026-03-02)
-------------------------------
Run the script and check printed output for:
- 2025 base population ~799,358 (or close, depending on parquet vintage)
- 2055 projection in 700,000-1,200,000 range
- Non-negative county populations
- Age distributions sum to total at each year

Inputs
------
- data/processed/base_population.parquet
    2025 base population by county (5-digit FIPS), single-year age, sex, race/ethnicity.
    53 counties x 91 ages x 2 sexes x 6 races. Source: Census PEP Vintage 2025.
- data/processed/sdc_2024/survival_rates_sdc_2024_full.csv
    CDC ND 2020 life table survival rates (annual and 5-year). 182 rows.
- data/processed/sdc_2024/fertility_rates_5yr_summary_sdc_2024.csv
    SDC 2018-2022 blended fertility rates. 7 age groups (15-49). Annual ASFRs.
- data/processed/migration/residual_migration_rates.parquet
    Residual migration rates by county, age group, sex, period. 5 periods (2000-2024).
    Annualized rates.
- data/raw/population/nd_county_population.csv
    County FIPS to county name mapping. 53 counties.
- data/raw/nd_sdc_2024_projections/state_projections.csv
    SDC 2024 original state projections for comparison.
- data/exports/baseline/summaries/state_total_population_by_year.csv
    Our full-engine baseline projections for comparison.

Output
------
- data/exports/sdc_method_new_data/state_population_by_year.csv
    State total population 2025-2055 in 5-year steps. 7 rows.
- data/exports/sdc_method_new_data/county_population_by_year.csv
    County population 2025-2055. 53 rows x 9 columns (fips, name, 7 years).
- data/exports/sdc_method_new_data/state_age_sex_by_year.csv
    State age-sex distribution 2025-2055. 252 rows (18 groups x 2 sexes x 7 years).

Usage
-----
    python scripts/exports/run_sdc_method_new_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 5-year age group bins (18 groups: 0-4 through 85+)
AGE_BINS = list(range(0, 85, 5)) + [85]  # [0, 5, 10, ..., 80, 85]
AGE_LABELS = [f"{a}-{a+4}" for a in range(0, 85, 5)] + ["85+"]
N_AGE_GROUPS = len(AGE_LABELS)  # 18

# Projection years
BASE_YEAR = 2025
END_YEAR = 2055
STEP = 5
PROJECTION_YEARS = list(range(BASE_YEAR, END_YEAR + 1, STEP))

# Bakken counties (5-digit FIPS) and dampening factor
BAKKEN_FIPS = {"38105", "38053", "38061", "38025", "38089"}
BAKKEN_DAMPENING = 0.6

# Sex ratio at birth
MALE_BIRTH_FRACTION = 0.512
FEMALE_BIRTH_FRACTION = 0.488

# Fertility age group boundaries (inclusive start, exclusive end for binning)
FERTILITY_AGE_STARTS = [15, 20, 25, 30, 35, 40, 45]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_base_population() -> pd.DataFrame:
    """Load and aggregate base population to 5-year age groups by county and sex.

    Returns DataFrame with columns: county_fips, age_group, sex, population
    """
    path = PROJECT_ROOT / "data" / "processed" / "base_population.parquet"
    df = pd.read_parquet(path)

    # Aggregate across race/ethnicity
    df = df.groupby(["county_fips", "age", "sex"], as_index=False)["population"].sum()

    # Bin single-year ages into 5-year groups
    # Ages 0-84 go into their respective 5-year bins; 85+ all go into "85+"
    df["age_group"] = pd.cut(
        df["age"],
        bins=AGE_BINS + [200],  # upper bound captures 85+
        labels=AGE_LABELS,
        right=False,
        include_lowest=True,
    )

    # Aggregate to 5-year groups
    result = (
        df.groupby(["county_fips", "age_group", "sex"], as_index=False, observed=True)[
            "population"
        ]
        .sum()
        .copy()
    )

    # Ensure age_group is string
    result["age_group"] = result["age_group"].astype(str)

    return result


def load_survival_rates() -> dict[tuple[str, str], float]:
    """Load 5-year survival rates keyed by (age_group, sex).

    Returns dict mapping (age_group_label, sex) -> survival_rate_5yr.
    One value per 5-year age group (uses the first single-year age in each group).
    """
    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "sdc_2024"
        / "survival_rates_sdc_2024_full.csv"
    )
    df = pd.read_csv(path)

    # Map single-year ages to 5-year group labels
    df["age_group"] = pd.cut(
        df["age"],
        bins=AGE_BINS + [200],
        labels=AGE_LABELS,
        right=False,
        include_lowest=True,
    ).astype(str)

    # Take one rate per group (they're all the same within a group)
    rates = (
        df.groupby(["age_group", "sex"], observed=True)["survival_rate_5yr"]
        .first()
        .to_dict()
    )

    return rates


def load_fertility_rates() -> dict[str, float]:
    """Load annual ASFRs keyed by age_group label (e.g., '15-19').

    Returns dict mapping age_group_label -> annual ASFR.
    """
    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "sdc_2024"
        / "fertility_rates_5yr_summary_sdc_2024.csv"
    )
    df = pd.read_csv(path)

    rates = {}
    for _, row in df.iterrows():
        label = f"{int(row['age_start'])}-{int(row['age_end'])}"
        rates[label] = row["asfr_annual"]

    return rates


def load_migration_rates() -> pd.DataFrame:
    """Load, convert, average, and dampen migration rates.

    Returns DataFrame with columns: county_fips, age_group, sex, migration_rate_5yr
    where migration_rate_5yr is the averaged and dampened 5-year rate.
    """
    path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "migration"
        / "residual_migration_rates.parquet"
    )
    df = pd.read_parquet(path)

    # Convert annualized rates to 5-year rates: (1+r)^5 - 1
    df["rate_5yr"] = (1 + df["migration_rate"]) ** 5 - 1

    # Simple arithmetic average across all periods
    avg_rates = (
        df.groupby(["county_fips", "age_group", "sex"], as_index=False)["rate_5yr"]
        .mean()
        .rename(columns={"rate_5yr": "migration_rate_5yr"})
    )

    # Apply flat 60% Bakken dampening
    bakken_mask = avg_rates["county_fips"].isin(BAKKEN_FIPS)
    avg_rates.loc[bakken_mask, "migration_rate_5yr"] *= BAKKEN_DAMPENING

    return avg_rates


def load_county_names() -> dict[str, str]:
    """Load county FIPS -> county name mapping.

    Returns dict mapping 5-digit FIPS string -> county name.
    """
    path = PROJECT_ROOT / "data" / "raw" / "population" / "nd_county_population.csv"
    df = pd.read_csv(path)
    return dict(zip(df["county_fips"].astype(str), df["county_name"]))


def load_sdc_2024_state() -> pd.DataFrame | None:
    """Load SDC 2024 original state projections for comparison."""
    path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "nd_sdc_2024_projections"
        / "state_projections.csv"
    )
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_our_baseline_state() -> pd.DataFrame | None:
    """Load our full-engine baseline state projections for comparison."""
    path = (
        PROJECT_ROOT
        / "data"
        / "exports"
        / "baseline"
        / "summaries"
        / "state_total_population_by_year.csv"
    )
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Projection Engine
# ---------------------------------------------------------------------------


def project_county(
    base_pop: pd.DataFrame,
    survival: dict[tuple[str, str], float],
    fertility: dict[str, float],
    mig_rates: pd.DataFrame,
    county_fips: str,
) -> list[dict]:
    """Run cohort-component projection for one county.

    Parameters
    ----------
    base_pop : DataFrame with columns county_fips, age_group, sex, population
    survival : dict (age_group, sex) -> survival_rate_5yr
    fertility : dict age_group -> annual ASFR
    mig_rates : DataFrame with columns county_fips, age_group, sex, migration_rate_5yr
    county_fips : 5-digit FIPS code string

    Returns
    -------
    List of dicts with keys: year, age_group, sex, population
    """
    # Filter to this county
    county_base = base_pop[base_pop["county_fips"] == county_fips].copy()
    county_mig = mig_rates[mig_rates["county_fips"] == county_fips].copy()

    # Build migration rate lookup: (age_group, sex) -> rate_5yr
    mig_lookup: dict[tuple[str, str], float] = {}
    for _, row in county_mig.iterrows():
        mig_lookup[(row["age_group"], row["sex"])] = row["migration_rate_5yr"]

    # Initialize population array: {(age_group, sex): population}
    pop: dict[tuple[str, str], float] = {}
    for _, row in county_base.iterrows():
        pop[(row["age_group"], row["sex"])] = row["population"]

    # Ensure all age-sex combinations exist
    for ag in AGE_LABELS:
        for sex in ["Male", "Female"]:
            if (ag, sex) not in pop:
                pop[(ag, sex)] = 0.0

    # Collect results
    results: list[dict] = []

    # Record base year
    for ag in AGE_LABELS:
        for sex in ["Male", "Female"]:
            results.append(
                {
                    "year": BASE_YEAR,
                    "age_group": ag,
                    "sex": sex,
                    "population": max(0.0, pop[(ag, sex)]),
                }
            )

    # Project forward
    for year in range(BASE_YEAR + STEP, END_YEAR + 1, STEP):
        new_pop: dict[tuple[str, str], float] = {}

        for sex in ["Male", "Female"]:
            # Step 1: SURVIVE ages 5-9 through 80-84
            # Each age group g receives survived population from the group 5 years younger
            for i in range(1, N_AGE_GROUPS - 1):  # indices 1 (5-9) through 16 (80-84)
                prev_ag = AGE_LABELS[i - 1]  # source age group (5 years younger)
                curr_ag = AGE_LABELS[i]  # destination age group
                surv = survival.get((prev_ag, sex), 0.0)
                new_pop[(curr_ag, sex)] = pop[(prev_ag, sex)] * surv

            # Step 2: OPEN-ENDED 85+
            surv_80_84 = survival.get(("80-84", sex), 0.0)
            surv_85p = survival.get(("85+", sex), 0.0)
            new_pop[("85+", sex)] = (
                pop[("80-84", sex)] * surv_80_84 + pop[("85+", sex)] * surv_85p
            )

            # Step 3: MIGRATE ages 5-9 through 85+
            for i in range(1, N_AGE_GROUPS):  # indices 1 (5-9) through 17 (85+)
                ag = AGE_LABELS[i]
                mig_rate = mig_lookup.get((ag, sex), 0.0)
                new_pop[(ag, sex)] *= 1 + mig_rate

        # Step 4: BIRTHS (use beginning-of-period female pop for simplicity)
        total_births_5yr = 0.0
        for fert_ag, asfr in fertility.items():
            fem_pop = pop.get((fert_ag, "Female"), 0.0)
            total_births_5yr += fem_pop * asfr * 5

        male_births = total_births_5yr * MALE_BIRTH_FRACTION
        female_births = total_births_5yr * FEMALE_BIRTH_FRACTION

        # Step 5: BIRTH SURVIVAL (0-4 group)
        surv_0_4_male = survival.get(("0-4", "Male"), 0.0)
        surv_0_4_female = survival.get(("0-4", "Female"), 0.0)
        new_pop[("0-4", "Male")] = male_births * surv_0_4_male
        new_pop[("0-4", "Female")] = female_births * surv_0_4_female

        # Step 6: MIGRATE 0-4
        for sex in ["Male", "Female"]:
            mig_rate = mig_lookup.get(("0-4", sex), 0.0)
            new_pop[("0-4", sex)] *= 1 + mig_rate

        # Floor at 0
        for key in new_pop:
            new_pop[key] = max(0.0, new_pop[key])

        # Record results
        for ag in AGE_LABELS:
            for sex in ["Male", "Female"]:
                results.append(
                    {
                        "year": year,
                        "age_group": ag,
                        "sex": sex,
                        "population": new_pop[(ag, sex)],
                    }
                )

        # Update pop for next iteration
        pop = new_pop

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run SDC methodology projection with latest data and write outputs."""
    print("=" * 72)
    print("SDC 2024 Method — Updated Data (2025 Base, 2000-2024 Migration)")
    print("=" * 72)

    # Load data
    print("\nLoading data...")
    base_pop = load_base_population()
    survival = load_survival_rates()
    fertility = load_fertility_rates()
    mig_rates = load_migration_rates()
    county_names = load_county_names()

    counties = sorted(base_pop["county_fips"].unique())
    print(f"  Counties: {len(counties)}")
    print(f"  Base population total: {base_pop['population'].sum():,.0f}")
    print(f"  Age groups: {N_AGE_GROUPS}")
    print(f"  Migration periods averaged: {mig_rates.shape[0] // (N_AGE_GROUPS * 2 * len(counties)) if len(counties) > 0 else 'N/A'}")
    print(f"  Bakken counties dampened: {len(BAKKEN_FIPS)}")
    print(f"  Fertility age groups: {len(fertility)}")

    # Run projections for all counties
    print("\nRunning projections...")
    all_results: list[dict] = []
    for fips in counties:
        county_results = project_county(base_pop, survival, fertility, mig_rates, fips)
        for r in county_results:
            r["county_fips"] = fips
        all_results.extend(county_results)

    results_df = pd.DataFrame(all_results)

    # ---------------------------------------------------------------------------
    # Output 1: State population by year
    # ---------------------------------------------------------------------------
    state_by_year = (
        results_df.groupby("year", as_index=False)["population"]
        .sum()
        .rename(columns={"population": "total_population"})
    )
    state_by_year["total_population"] = state_by_year["total_population"].round(0).astype(int)

    # ---------------------------------------------------------------------------
    # Output 2: County population by year
    # ---------------------------------------------------------------------------
    county_totals = (
        results_df.groupby(["county_fips", "year"], as_index=False)["population"].sum()
    )
    county_pivot = county_totals.pivot(
        index="county_fips", columns="year", values="population"
    ).reset_index()
    county_pivot.columns = ["county_fips"] + [
        str(int(c)) for c in county_pivot.columns[1:]
    ]
    county_pivot.insert(
        1,
        "county_name",
        county_pivot["county_fips"].map(county_names),
    )
    # Round to integers
    for col in county_pivot.columns[2:]:
        county_pivot[col] = county_pivot[col].round(0).astype(int)
    county_pivot = county_pivot.sort_values("county_fips").reset_index(drop=True)

    # ---------------------------------------------------------------------------
    # Output 3: State age-sex by year
    # ---------------------------------------------------------------------------
    state_age_sex = (
        results_df.groupby(["year", "age_group", "sex"], as_index=False)["population"]
        .sum()
    )
    state_age_sex["population"] = state_age_sex["population"].round(0).astype(int)

    # Sort by year, then by age group order, then by sex
    age_order = {ag: i for i, ag in enumerate(AGE_LABELS)}
    state_age_sex["_age_order"] = state_age_sex["age_group"].map(age_order)
    sex_order = {"Male": 0, "Female": 1}
    state_age_sex["_sex_order"] = state_age_sex["sex"].map(sex_order)
    state_age_sex = state_age_sex.sort_values(
        ["year", "_age_order", "_sex_order"]
    ).drop(columns=["_age_order", "_sex_order"])

    # ---------------------------------------------------------------------------
    # Write outputs
    # ---------------------------------------------------------------------------
    out_dir = PROJECT_ROOT / "data" / "exports" / "sdc_method_new_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / "state_population_by_year.csv"
    state_by_year.to_csv(state_path, index=False)
    print(f"\nWrote: {state_path.relative_to(PROJECT_ROOT)}")

    county_path = out_dir / "county_population_by_year.csv"
    county_pivot.to_csv(county_path, index=False)
    print(f"Wrote: {county_path.relative_to(PROJECT_ROOT)}")

    age_sex_path = out_dir / "state_age_sex_by_year.csv"
    state_age_sex.to_csv(age_sex_path, index=False)
    print(f"Wrote: {age_sex_path.relative_to(PROJECT_ROOT)}")

    # ---------------------------------------------------------------------------
    # Summary output
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("STATE TOTAL POPULATION BY YEAR")
    print("=" * 72)
    for _, row in state_by_year.iterrows():
        print(f"  {int(row['year'])}: {row['total_population']:>10,}")

    base = state_by_year[state_by_year["year"] == BASE_YEAR]["total_population"].iloc[0]
    final = state_by_year[state_by_year["year"] == END_YEAR]["total_population"].iloc[0]
    pct_change = (final - base) / base * 100
    print(f"\n  Change 2025-2055: {final - base:+,} ({pct_change:+.1f}%)")

    # ---------------------------------------------------------------------------
    # Comparison with SDC 2024 original
    # ---------------------------------------------------------------------------
    sdc_orig = load_sdc_2024_state()
    if sdc_orig is not None:
        print("\n" + "-" * 72)
        print("COMPARISON: SDC 2024 Original vs. SDC Method + New Data")
        print("-" * 72)
        print(f"  {'Year':>6}  {'SDC 2024':>12}  {'New Data':>12}  {'Diff':>10}  {'% Diff':>8}")
        for _, row in state_by_year.iterrows():
            yr = int(row["year"])
            new_val = row["total_population"]
            sdc_row = sdc_orig[sdc_orig["year"] == yr]
            if len(sdc_row) > 0:
                sdc_val = int(sdc_row["total_population"].iloc[0])
                diff = new_val - sdc_val
                pct = diff / sdc_val * 100
                print(f"  {yr:>6}  {sdc_val:>12,}  {new_val:>12,}  {diff:>+10,}  {pct:>+7.1f}%")
            else:
                print(f"  {yr:>6}  {'N/A':>12}  {new_val:>12,}")

    # ---------------------------------------------------------------------------
    # Comparison with our baseline
    # ---------------------------------------------------------------------------
    our_baseline = load_our_baseline_state()
    if our_baseline is not None:
        print("\n" + "-" * 72)
        print("COMPARISON: Our Baseline (Full Engine) vs. SDC Method + New Data")
        print("-" * 72)
        # Our baseline has annual data and the column is named differently
        # Find the year columns
        year_cols = [c for c in our_baseline.columns if c not in ("fips",)]
        print(f"  {'Year':>6}  {'Baseline':>12}  {'SDC+NewData':>12}  {'Diff':>10}  {'% Diff':>8}")
        for _, row in state_by_year.iterrows():
            yr = int(row["year"])
            new_val = row["total_population"]
            yr_str = str(yr)
            if yr_str in our_baseline.columns:
                bl_val = int(round(our_baseline[yr_str].iloc[0]))
                diff = new_val - bl_val
                pct = diff / bl_val * 100
                print(f"  {yr:>6}  {bl_val:>12,}  {new_val:>12,}  {diff:>+10,}  {pct:>+7.1f}%")
            else:
                print(f"  {yr:>6}  {'N/A':>12}  {new_val:>12,}")

    # ---------------------------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("SANITY CHECKS")
    print("-" * 72)

    # Check 2025 base
    print(f"  2025 base population: {base:,} (expected ~799,358)")

    # Check final year range
    in_range = 700_000 <= final <= 1_200_000
    print(f"  2055 projection: {final:,} (in 700k-1.2M range: {'YES' if in_range else 'NO !!!'})")

    # Check negative populations
    neg_count = (results_df["population"] < 0).sum()
    print(f"  Negative populations: {neg_count}")

    # Check age sums match totals
    for yr in PROJECTION_YEARS:
        yr_data = results_df[results_df["year"] == yr]
        age_sum = yr_data["population"].sum()
        total = state_by_year[state_by_year["year"] == yr]["total_population"].iloc[0]
        delta = abs(age_sum - total)
        status = "OK" if delta < 1 else f"MISMATCH ({delta:.0f})"
        if yr == BASE_YEAR or yr == END_YEAR:
            print(f"  Age-sex sum {yr}: {age_sum:,.0f} vs total {total:,} — {status}")

    # Top 5 / Bottom 5 counties by 2055
    print("\n  Top 5 counties by 2055 population:")
    top5 = county_pivot.nlargest(5, "2055")
    for _, r in top5.iterrows():
        print(f"    {r['county_name']:>15}: {r['2055']:>10,}")

    print("\n  Bottom 5 counties by 2055 population:")
    bot5 = county_pivot.nsmallest(5, "2055")
    for _, r in bot5.iterrows():
        print(f"    {r['county_name']:>15}: {r['2055']:>10,}")

    print("\n" + "=" * 72)
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()
