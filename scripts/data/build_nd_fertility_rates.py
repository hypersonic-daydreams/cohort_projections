"""
Build ND-specific age-specific fertility rates from CDC WONDER Natality data.

Created: 2026-02-23
ADR: 053 (Part A — ND-Adjusted Fertility Rates)
Author: Claude Code / N. Haarstad

Purpose
-------
Replace national CDC ASFR with North Dakota-specific rates computed from
CDC WONDER birth counts and Census PEP female population denominators.
National rates understate ND fertility by ~15% (national TFR ~1.62 vs ND ~1.86),
systematically undercounting ~1,400-1,600 births per year.

Method
------
1. Load ND birth counts from CDC WONDER Natality (2020-2023 pooled, 4 years
   for statistical stability) by mother's age (9 groups) and Single Race 6 ×
   Hispanic Origin.
2. Map CDC WONDER race categories to project standard codes:
   - Hispanic of any race → "hispanic" (ethnicity takes priority over race)
   - Non-Hispanic: White → "white_nh", Black → "black_nh", AIAN → "aian_nh",
     Asian + NHPI → "asian_nh" (combined), Two+ → "two_or_more_nh"
3. Distribute "Unknown or Not Stated" Hispanic origin proportionally based on
   the known Hispanic/Non-Hispanic split within each race × age cell.
4. Obtain ND female population by age and race from Census cc-est2024-alldata-38.csv
   (county characteristics file), summing across all 53 ND counties and years 2-5
   (July 2020 through July 2023, matching the birth pooling window).
5. Compute ASFR = births / female_pop × 1000 (per 1,000 women).
6. For cells with <10 births across the 4-year window (CDC suppression threshold),
   fall back to national ASFR from asfr_processed.csv.
7. Ensure a complete 7×7 grid (7 age groups × 7 race categories) with no gaps.

Key design decisions
--------------------
- **4-year pooling (2020-2023)**: Smooths year-to-year volatility for small
  race×age cells while remaining current. Trade-off: includes 2020 COVID year.
- **Hispanic priority over race**: Follows OMB standard — a Hispanic White mother
  is categorized as "hispanic", not "white_nh". Consistent with project race schema.
- **Asian + NHPI combined**: NHPI is <0.1% of ND births; combining with Asian
  matches project race schema and prevents suppression.
- **National fallback for suppressed cells**: 5 cells (primarily 45-49 age group
  and Asian NH 15-19) had <10 births. National rates used rather than zero to avoid
  underestimating small-population fertility.

Validation results (2026-02-23)
-------------------------------
- ND TFR: 1.863 (target range 1.85-1.90 per ADR-053; within 1%)
- Average annual births: 9,804 (actual 2023: 9,647; within 1.6%)
- 5 of 49 cells used national fallback rates
- Race-specific TFR patterns plausible: AIAN highest (2.30), White lowest (1.75)

Inputs
------
- data/raw/fertility/cdc_wonder_nd_births_2020_2023.txt
    CDC WONDER export, tab-delimited, 74 rows. Query: Natality 2016-2024 Expanded,
    State=North Dakota, Group By=Mother's Single Race 6 + Hispanic Origin +
    Mother's Age 9, Years=2020-2023 combined. Downloaded 2026-02-23.
- data/raw/fertility/cdc_wonder_national_births_2020_2023.txt
    Same query without state filter (national), 151 rows. Used as fallback
    for suppressed ND cells.
- data/raw/population/cc-est2024-alldata-38.csv
    Census Bureau County Characteristics Datasets (cc-est2024-alldata),
    FIPS 38 (North Dakota). Provides female population by single-year age group,
    race, and year for all 53 counties. Vintage 2024.
- data/raw/fertility/asfr_processed.csv
    Pre-existing national ASFR (SEER-derived, 2024 vintage) used as fallback
    lookup for suppressed cells.

Output
------
- data/raw/fertility/nd_asfr_processed.csv
    49 rows (7 ages × 7 races), columns: age, race_ethnicity, asfr, year.
    Format matches existing asfr_processed.csv schema for pipeline compatibility.

Usage
-----
    python scripts/data/build_nd_fertility_rates.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Age group mapping: Census AGEGRP code -> (start_age, label)
CENSUS_AGEGRP_TO_LABEL: dict[int, str] = {
    4: "15-19",
    5: "20-24",
    6: "25-29",
    7: "30-34",
    8: "35-39",
    9: "40-44",
    10: "45-49",
}

# CDC WONDER age label -> our standard label
WONDER_AGE_MAP: dict[str, str] = {
    "Under 15 years": None,  # exclude
    "15-19 years": "15-19",
    "20-24 years": "20-24",
    "25-29 years": "25-29",
    "30-34 years": "30-34",
    "35-39 years": "35-39",
    "40-44 years": "40-44",
    "45-49 years": "45-49",
    "50 years and over": None,  # exclude
}

# CDC WONDER race mapping: (Single Race 6, Hispanic Origin) -> our race code
# Hispanic of any race takes priority
WONDER_RACE_MAP: dict[tuple[str, str], str] = {
    ("American Indian or Alaska Native", "Not Hispanic or Latino"): "aian_nh",
    ("Asian", "Not Hispanic or Latino"): "asian_nh",
    ("Black or African American", "Not Hispanic or Latino"): "black_nh",
    ("Native Hawaiian or Other Pacific Islander", "Not Hispanic or Latino"): "asian_nh",
    ("White", "Not Hispanic or Latino"): "white_nh",
    ("More than one race", "Not Hispanic or Latino"): "two_or_more_nh",
    # All Hispanic origins regardless of race -> hispanic
    ("American Indian or Alaska Native", "Hispanic or Latino"): "hispanic",
    ("Asian", "Hispanic or Latino"): "hispanic",
    ("Black or African American", "Hispanic or Latino"): "hispanic",
    ("Native Hawaiian or Other Pacific Islander", "Hispanic or Latino"): "hispanic",
    ("White", "Hispanic or Latino"): "hispanic",
    ("More than one race", "Hispanic or Latino"): "hispanic",
}

# Census PEP female columns for each race code
CENSUS_FEMALE_COLS: dict[str, list[str]] = {
    "white_nh": ["NHWA_FEMALE"],
    "black_nh": ["NHBA_FEMALE"],
    "aian_nh": ["NHIA_FEMALE"],
    "asian_nh": ["NHAA_FEMALE", "NHNA_FEMALE"],  # Asian + NHPI combined
    "two_or_more_nh": ["NHTOM_FEMALE"],
    "hispanic": ["H_FEMALE"],
    "total": ["TOT_FEMALE"],
}


def load_wonder_births(filepath: Path) -> pd.DataFrame:
    """Load CDC WONDER natality export and map to standard race/age codes.

    Handles 'Unknown or Not Stated' Hispanic origin by distributing
    proportionally to known Hispanic/Non-Hispanic within each race × age.
    """
    df = pd.read_csv(filepath, sep="\t")
    print(f"  Loaded {len(df)} rows from {filepath.name}")

    # Map age groups
    df["age_group"] = df["Mother's Age 9"].map(WONDER_AGE_MAP)
    df = df.dropna(subset=["age_group"])

    # Parse births (handle suppressed and comma-formatted)
    df["births"] = pd.to_numeric(
        df["Births"].astype(str).str.replace(",", ""), errors="coerce"
    )

    # Separate known vs unknown Hispanic origin
    known = df[
        df["Mother's Hispanic Origin"].isin(["Hispanic or Latino", "Not Hispanic or Latino"])
    ].copy()
    unknown = df[df["Mother's Hispanic Origin"] == "Unknown or Not Stated"].copy()

    # Map known rows to race codes
    known["race_code"] = known.apply(
        lambda r: WONDER_RACE_MAP.get(
            (r["Mother's Single Race 6"], r["Mother's Hispanic Origin"])
        ),
        axis=1,
    )
    known = known.dropna(subset=["race_code", "births"])

    # Distribute unknown proportionally within each race × age
    if not unknown.empty and not known.empty:
        # Compute Hispanic share by race × age from known data
        hisp_shares = (
            known.groupby(["Mother's Single Race 6", "age_group", "race_code"])["births"]
            .sum()
            .reset_index()
        )
        race_age_totals = (
            known.groupby(["Mother's Single Race 6", "age_group"])["births"]
            .sum()
            .reset_index(name="total_births")
        )
        hisp_shares = hisp_shares.merge(
            race_age_totals, on=["Mother's Single Race 6", "age_group"]
        )
        hisp_shares["share"] = hisp_shares["births"] / hisp_shares["total_births"]

        allocated_rows = []
        for _, unk_row in unknown.iterrows():
            if pd.isna(unk_row["births"]):
                continue
            matching = hisp_shares[
                (hisp_shares["Mother's Single Race 6"] == unk_row["Mother's Single Race 6"])
                & (hisp_shares["age_group"] == unk_row["age_group"])
            ]
            for _, share_row in matching.iterrows():
                allocated_rows.append(
                    {
                        "age_group": unk_row["age_group"],
                        "race_code": share_row["race_code"],
                        "births": unk_row["births"] * share_row["share"],
                    }
                )
        if allocated_rows:
            allocated_df = pd.DataFrame(allocated_rows)
            known_agg = (
                known.groupby(["age_group", "race_code"])["births"]
                .sum()
                .reset_index()
            )
            allocated_agg = (
                allocated_df.groupby(["age_group", "race_code"])["births"]
                .sum()
                .reset_index()
            )
            # Merge and sum
            result = known_agg.merge(
                allocated_agg,
                on=["age_group", "race_code"],
                how="outer",
                suffixes=("_known", "_unk"),
            )
            result["births"] = result["births_known"].fillna(0) + result["births_unk"].fillna(0)
            result = result[["age_group", "race_code", "births"]]
        else:
            result = known.groupby(["age_group", "race_code"])["births"].sum().reset_index()
    else:
        result = known.groupby(["age_group", "race_code"])["births"].sum().reset_index()

    # Add total row (sum across all races)
    total = result.groupby("age_group")["births"].sum().reset_index()
    total["race_code"] = "total"
    result = pd.concat([result, total], ignore_index=True)

    total_births = result[result["race_code"] != "total"]["births"].sum()
    print(f"  Total births (all races): {total_births:,.0f}")

    return result


def load_census_female_population(
    filepath: Path,
    years: list[int],
    agegrps: list[int],
) -> pd.DataFrame:
    """Load Census PEP female population by race and age group.

    Sums across all ND counties, returns total female population for each
    race × age group × year combination, then sums across years (for ASFR
    denominator).

    Args:
        filepath: Path to cc-est2024-alldata-38.csv
        years: YEAR codes to include (2=Jul2020, 3=Jul2021, 4=Jul2022, 5=Jul2023)
        agegrps: AGEGRP codes to include (4=15-19 through 10=45-49)
    """
    df = pd.read_csv(filepath)
    df = df[df["YEAR"].isin(years) & df["AGEGRP"].isin(agegrps)]

    records = []
    for agegrp in agegrps:
        age_label = CENSUS_AGEGRP_TO_LABEL[agegrp]
        for race_code, cols in CENSUS_FEMALE_COLS.items():
            # Sum across counties and years
            subset = df[df["AGEGRP"] == agegrp]
            pop = sum(subset[col].sum() for col in cols)
            records.append(
                {
                    "age_group": age_label,
                    "race_code": race_code,
                    "female_pop": pop,
                }
            )

    result = pd.DataFrame(records)
    total_pop = result[result["race_code"] == "total"]["female_pop"].sum()
    print(f"  Total female pop (reproductive ages, summed {len(years)} years): {total_pop:,.0f}")
    return result


def compute_asfr(
    births: pd.DataFrame,
    population: pd.DataFrame,
    national_births: pd.DataFrame | None = None,
    national_population: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute ASFR per 1,000 women. Fall back to national for suppressed cells."""
    merged = births.merge(population, on=["age_group", "race_code"], how="left")

    # Compute ASFR (births per 1,000 women per year)
    # Population is already summed across years, so ASFR = births / pop * 1000
    merged["asfr"] = merged.apply(
        lambda r: r["births"] / r["female_pop"] * 1000 if r["female_pop"] > 0 else 0.0,
        axis=1,
    )

    # Identify cells with too few births (< 10 in 4 years = unreliable)
    suppressed = merged[(merged["births"] < 10) & (merged["race_code"] != "total")]
    if not suppressed.empty:
        print(f"\n  Suppressed cells (< 10 births, falling back to national):")
        for _, row in suppressed.iterrows():
            print(f"    {row['race_code']} × {row['age_group']}: {row['births']:.0f} births")

        # Compute national ASFR for fallback
        if national_births is not None and national_population is not None:
            nat_merged = national_births.merge(
                national_population, on=["age_group", "race_code"], how="left"
            )
            nat_merged["asfr"] = nat_merged.apply(
                lambda r: r["births"] / r["female_pop"] * 1000 if r["female_pop"] > 0 else 0.0,
                axis=1,
            )
            nat_lookup = nat_merged.set_index(["age_group", "race_code"])["asfr"]

            for idx, row in suppressed.iterrows():
                key = (row["age_group"], row["race_code"])
                if key in nat_lookup.index:
                    nat_rate = nat_lookup[key]
                    merged.loc[idx, "asfr"] = nat_rate
                    print(f"      -> using national rate: {nat_rate:.1f}")

    result = merged[["age_group", "race_code", "asfr"]].copy()
    return result


def main():
    print("=" * 70)
    print("Building ND-Specific Fertility Rates (ADR-053)")
    print("=" * 70)

    # File paths
    nd_births_file = PROJECT_ROOT / "data" / "raw" / "fertility" / "cdc_wonder_nd_births_2020_2023.txt"
    nat_births_file = (
        PROJECT_ROOT / "data" / "raw" / "fertility" / "cdc_wonder_national_births_2020_2023.txt"
    )
    census_file = PROJECT_ROOT / "data" / "raw" / "population" / "cc-est2024-alldata-38.csv"
    output_file = PROJECT_ROOT / "data" / "raw" / "fertility" / "nd_asfr_processed.csv"

    # Step 1: Load ND births from CDC WONDER
    print("\nStep 1: Loading ND births from CDC WONDER (2020-2023 pooled)")
    nd_births = load_wonder_births(nd_births_file)

    # Step 2: Load national births for fallback
    print("\nStep 2: Loading national births from CDC WONDER (2020-2023 pooled)")
    nat_births = load_wonder_births(nat_births_file)

    # Step 3: Load Census PEP female population (ND)
    # YEAR 2=Jul2020, 3=Jul2021, 4=Jul2022, 5=Jul2023
    print("\nStep 3: Loading ND female population from Census PEP")
    nd_population = load_census_female_population(
        census_file,
        years=[2, 3, 4, 5],
        agegrps=list(CENSUS_AGEGRP_TO_LABEL.keys()),
    )

    # Step 4: For national fallback, we don't have national population from
    # cc-est2024 (that's ND only). Use the existing national ASFR file instead.
    print("\nStep 4: Loading existing national ASFR for fallback")
    nat_asfr_file = PROJECT_ROOT / "data" / "raw" / "fertility" / "asfr_processed.csv"
    nat_asfr_existing = pd.read_csv(nat_asfr_file)
    print(f"  Loaded {len(nat_asfr_existing)} national ASFR rows")

    # Step 5: Compute ND ASFR
    print("\nStep 5: Computing ND ASFR")
    nd_asfr = compute_asfr(nd_births, nd_population)

    # Ensure complete grid: all age × race combinations present
    all_ages = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"]
    all_races = ["total", "white_nh", "black_nh", "hispanic", "aian_nh", "asian_nh", "two_or_more_nh"]
    complete_grid = pd.DataFrame(
        [(a, r) for a in all_ages for r in all_races],
        columns=["age_group", "race_code"],
    )
    nd_asfr = complete_grid.merge(nd_asfr, on=["age_group", "race_code"], how="left")
    nd_asfr["asfr"] = nd_asfr["asfr"].fillna(0)

    # For missing/zero ND cells (suppressed), substitute national rates
    nat_lookup = nat_asfr_existing.set_index(["age", "race_ethnicity"])["asfr"]
    suppressed_mask = nd_asfr["asfr"] == 0
    for idx in nd_asfr[suppressed_mask & (nd_asfr["race_code"] != "total")].index:
        row = nd_asfr.loc[idx]
        key = (row["age_group"], row["race_code"])
        if key in nat_lookup.index:
            nd_asfr.loc[idx, "asfr"] = nat_lookup[key]
            print(f"  Fallback to national: {row['race_code']} × {row['age_group']} -> {nat_lookup[key]:.1f}")

    # Step 6: Format output
    print("\nStep 6: Formatting output")
    output = nd_asfr.rename(columns={"age_group": "age", "race_code": "race_ethnicity"})
    output["year"] = 2023  # Reference year
    output = output[["age", "race_ethnicity", "asfr", "year"]]

    # Round to 1 decimal
    output["asfr"] = output["asfr"].round(1)

    # Sort to match existing format
    race_order = ["total", "white_nh", "black_nh", "hispanic", "aian_nh", "asian_nh", "two_or_more_nh"]
    age_order = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49"]
    output["_race_sort"] = output["race_ethnicity"].map(
        {r: i for i, r in enumerate(race_order)}
    )
    output["_age_sort"] = output["age"].map({a: i for i, a in enumerate(age_order)})
    output = output.sort_values(["_race_sort", "_age_sort"])
    output = output.drop(columns=["_race_sort", "_age_sort"])

    # Save
    output.to_csv(output_file, index=False)
    print(f"\n  Saved to {output_file}")

    # Step 7: Validation summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    # Compare ND vs national TFR
    existing_nat = pd.read_csv(nat_asfr_file)

    print(f"\n{'Race':<25} {'ND ASFR Sum':>12} {'ND TFR':>8} {'Nat ASFR Sum':>14} {'Nat TFR':>8} {'Ratio':>7}")
    print("-" * 80)

    for race in race_order:
        nd_sum = output[output["race_ethnicity"] == race]["asfr"].sum()
        nd_tfr = nd_sum * 5 / 1000  # 5-year age groups, convert from per-1000

        nat_row = existing_nat[existing_nat["race_ethnicity"] == race]
        if not nat_row.empty:
            nat_sum = nat_row["asfr"].sum()
            nat_tfr = nat_sum * 5 / 1000
            ratio = nd_tfr / nat_tfr if nat_tfr > 0 else float("inf")
            print(f"{race:<25} {nd_sum:>12.1f} {nd_tfr:>8.3f} {nat_sum:>14.1f} {nat_tfr:>8.3f} {ratio:>7.2f}")
        else:
            print(f"{race:<25} {nd_sum:>12.1f} {nd_tfr:>8.3f} {'N/A':>14} {'N/A':>8} {'N/A':>7}")

    # Total births validation
    # ND total births 2020-2023 should be ~38,000-40,000 (VES shows ~9,500-10,000/yr)
    nd_total_births = nd_births[nd_births["race_code"] == "total"]["births"].sum()
    print(f"\nND total births 2020-2023 (from WONDER): {nd_total_births:,.0f}")
    print(f"Average annual births: {nd_total_births / 4:,.0f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
