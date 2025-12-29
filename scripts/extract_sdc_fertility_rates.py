#!/usr/bin/env python3
"""
Extract fertility rates from SDC 2024 source files.

This script reads the SDC 2024 Excel workbooks containing:
1. Birth data (2018-2022) by age group
2. Female population (2018-2022 average) by age group
3. Calculated fertility rates from the projections workbook

SDC Methodology (per README.md):
- Fertility rates based on ND DHHS Vital Statistics 2016-2022
- Blended with state and national rates (CDC NVSS 2021)
- Rates smoothed to reduce anomalies
- 5-year average fertility from 2018-2022

Outputs:
1. fertility_rates_sdc_2024.csv - Single-year ages (15-49) interpolated from raw data
2. fertility_rates_5yr_summary_sdc_2024.csv - 5-year group summary with calculated ASFR
3. fertility_rates_sdc_blended_2024.csv - SDC's blended/smoothed rates (as used in projections)

Note: SDC rates differ from raw calculation because they blend with national rates
and apply smoothing to reduce county-level anomalies.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate

# File paths
BASE_DIR = Path("/home/nigel/cohort_projections")
DATA_DIR = BASE_DIR / "data/raw/nd_sdc_2024_projections/source_files"
OUTPUT_DIR = BASE_DIR / "data/processed/sdc_2024"

BIRTH_FILE = DATA_DIR / "fertility/Copy of 2018-2022 ND Res Birth for Kevin Iverson.xlsx"
FEMALE_POP_FILE = DATA_DIR / "fertility/Average Female Count 2018 to 2022.xlsx"
PROJECTIONS_FILE = DATA_DIR / "results/Projections_Base_2023.xlsx"


def extract_births_by_age_group() -> pd.DataFrame:
    """Extract births by age group from the birth data file."""
    df = pd.read_excel(BIRTH_FILE, sheet_name="Sheet3", header=None)

    # State total births by age group (column 1 is North Dakota totals)
    # Rows 3-9 contain age groups, column 0 is age label, column 1 is state total
    births_data = []
    age_groups = {
        "UNDER 20": (15, 19),  # SDC uses "UNDER 20" for 15-19
        "20 TO 24": (20, 24),
        "25 TO 29": (25, 29),
        "30 TO 34": (30, 34),
        "35 TO 39": (35, 39),
        "40 TO 44": (40, 44),
        "45+": (45, 49),  # SDC uses "45+" for 45-49
    }

    for i in range(3, 10):
        age_label = df.iloc[i, 0]
        births = df.iloc[i, 1]
        if age_label in age_groups:
            age_start, age_end = age_groups[age_label]
            births_data.append(
                {
                    "age_group": age_label,
                    "age_start": age_start,
                    "age_end": age_end,
                    "births_5yr": births,  # Total births 2018-2022
                }
            )

    return pd.DataFrame(births_data)


def extract_female_population() -> pd.DataFrame:
    """Extract average female population by age group."""
    df = pd.read_excel(FEMALE_POP_FILE, sheet_name="Sheet1", header=None)

    # Find the Average section starting around row 24-25
    # Column 3 contains North Dakota values
    pop_data = []
    age_mapping = {
        "AGE1014_FEM": (10, 14),  # Note: We'll need to split this for 15-19
        "AGE1519_FEM": (15, 19),
        "AGE2024_FEM": (20, 24),
        "AGE2529_FEM": (25, 29),
        "AGE3034_FEM": (30, 34),
        "AGE3539_FEM": (35, 39),
        "AGE4044_FEM": (40, 44),
        "AGE4549_FEM": (45, 49),
    }

    # Find rows with average data (around row 26-33)
    for i in range(26, 34):
        age_label = df.iloc[i, 1]
        pop = df.iloc[i, 3]  # North Dakota column
        if age_label in age_mapping:
            age_start, age_end = age_mapping[age_label]
            pop_data.append(
                {
                    "age_group": age_label,
                    "age_start": age_start,
                    "age_end": age_end,
                    "avg_female_pop": pop,  # Average 2018-2022
                }
            )

    return pd.DataFrame(pop_data)


def extract_sdc_rates_from_projections() -> pd.DataFrame:
    """
    Extract the calculated fertility rates from SDC projections workbook.

    These are SDC's blended/smoothed rates that combine:
    - ND vital statistics 2016-2022
    - CDC NVSS 2021 national rates
    - Smoothing to reduce county-level anomalies

    The rates in the projections file are 5-year cumulative rates
    (births per woman over a 5-year period).
    """
    df = pd.read_excel(PROJECTIONS_FILE, sheet_name="Fer 2020 - 2025", header=None)

    # The 5-year fertility rates are in rows 38-45, column 3 (North Dakota)
    # These are: births per woman over the 5-year period by 5-year age group
    rates_data = []
    age_mapping = {
        "10-14": (10, 14),
        "15-19": (15, 19),
        "20-24": (20, 24),
        "25-29": (25, 29),
        "30-34": (30, 34),
        "35-39": (35, 39),
        "40-44": (40, 44),
        "45-49": (45, 49),
    }

    for i in range(38, 46):
        age_label = df.iloc[i, 2]
        rate = df.iloc[i, 3]  # North Dakota column
        if age_label in age_mapping:
            age_start, age_end = age_mapping[age_label]
            rates_data.append(
                {
                    "age_group": age_label,
                    "age_start": age_start,
                    "age_end": age_end,
                    "fertility_rate_5yr": rate,  # 5-year cumulative rate
                }
            )

    return pd.DataFrame(rates_data)


def interpolate_sdc_blended_rates(sdc_rates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate SDC's blended/smoothed 5-year rates to single-year ages.

    These are the rates SDC actually uses in their projections, which include
    blending with national rates and smoothing.
    """
    # Filter to reproductive ages (15-49) only
    sdc_repro = sdc_rates_df[sdc_rates_df["age_start"] >= 15].copy()

    # Convert 5-year cumulative rate to annual rate
    sdc_repro["asfr_annual"] = sdc_repro["fertility_rate_5yr"] / 5

    # Use midpoint of each age group for interpolation
    midpoints = (sdc_repro["age_start"] + sdc_repro["age_end"]) / 2
    rates = sdc_repro["asfr_annual"].values

    # Create cubic spline interpolation
    spline = interpolate.CubicSpline(midpoints, rates, bc_type="natural")

    # Interpolate for single years 15-49
    single_years = np.arange(15, 50)
    interpolated_rates = spline(single_years)

    # Ensure no negative rates
    interpolated_rates = np.maximum(interpolated_rates, 0)

    return pd.DataFrame(
        {
            "age": single_years,
            "fertility_rate": interpolated_rates,
            "source": "SDC_2024_blended",
            "notes": "SDC blended rates (ND + national); interpolated using cubic spline",
        }
    )


def calculate_asfr_from_raw_data(births_df: pd.DataFrame, pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Age-Specific Fertility Rates from raw births and population data.

    ASFR = (births in 5 years) / (avg female population × 5 years)
         = births per woman per year for each 5-year age group
    """
    # Merge births and population
    # Need to handle age group naming differences
    # SDC birth data uses: UNDER 20, 20 TO 24, etc.
    # SDC pop data uses: AGE1519_FEM, AGE2024_FEM, etc.

    # We'll calculate based on the 5-year groups that match reproductive ages (15-49)

    # For reproductive ages 15-49, we need populations for ages 15-19, 20-24, ..., 45-49
    # Birth data starts with "UNDER 20" which we'll treat as 15-19

    # Create lookup by age range
    pop_lookup = {row["age_start"]: row["avg_female_pop"] for _, row in pop_df.iterrows()}

    asfr_data = []
    for _, row in births_df.iterrows():
        age_start = row["age_start"]
        if age_start in pop_lookup:
            # ASFR = births / (population × years)
            # births_5yr is total births over 5 years (2018-2022)
            # avg_female_pop is average over those 5 years
            # ASFR = births_5yr / (avg_female_pop × 5)
            asfr = row["births_5yr"] / (pop_lookup[age_start] * 5)
            asfr_data.append(
                {
                    "age_start": age_start,
                    "age_end": row["age_end"],
                    "births_5yr": row["births_5yr"],
                    "avg_female_pop": pop_lookup[age_start],
                    "asfr": asfr,  # Annual births per woman
                }
            )

    return pd.DataFrame(asfr_data)


def interpolate_to_single_years(asfr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate 5-year ASFRs to single-year ages using cubic spline.

    Uses midpoints of age groups for interpolation anchor points.
    """
    # Use midpoint of each age group for interpolation
    midpoints = (asfr_df["age_start"] + asfr_df["age_end"]) / 2
    rates = asfr_df["asfr"].values

    # Create cubic spline interpolation
    # Use natural boundary conditions to avoid edge effects
    spline = interpolate.CubicSpline(midpoints, rates, bc_type="natural")

    # Interpolate for single years 15-49
    single_years = np.arange(15, 50)
    interpolated_rates = spline(single_years)

    # Ensure no negative rates (can happen at edges)
    interpolated_rates = np.maximum(interpolated_rates, 0)

    # Create output dataframe
    result = pd.DataFrame({"age": single_years, "fertility_rate": interpolated_rates})

    return result


def main():
    print("=" * 60)
    print("SDC 2024 Fertility Rate Extraction")
    print("=" * 60)

    # Step 1: Extract raw data
    print("\n1. Extracting births by age group...")
    births_df = extract_births_by_age_group()
    print(births_df.to_string(index=False))

    print("\n2. Extracting female population by age group...")
    pop_df = extract_female_population()
    print(pop_df.to_string(index=False))

    print("\n3. Extracting SDC calculated rates from projections workbook...")
    sdc_rates_df = extract_sdc_rates_from_projections()
    print(sdc_rates_df.to_string(index=False))

    # Step 2: Calculate ASFR from raw data
    print("\n4. Calculating ASFR from raw births and population...")
    asfr_df = calculate_asfr_from_raw_data(births_df, pop_df)
    print(asfr_df.to_string(index=False))

    # Compare with SDC rates
    print("\n5. Comparison with SDC 5-year rates:")
    print("   (SDC rates are cumulative over 5 years, ASFR is annual)")
    for _, row in asfr_df.iterrows():
        sdc_5yr = sdc_rates_df[sdc_rates_df["age_start"] == row["age_start"]][
            "fertility_rate_5yr"
        ].values
        if len(sdc_5yr) > 0:
            # SDC 5-year rate should equal ASFR × 5
            print(
                f"   Age {int(row['age_start'])}-{int(row['age_end'])}: "
                f"ASFR×5 = {row['asfr']*5:.6f}, SDC 5yr = {sdc_5yr[0]:.6f}"
            )

    # Step 3: Interpolate to single years (from raw data)
    print("\n6. Interpolating raw data to single-year ages (15-49)...")
    single_year_df = interpolate_to_single_years(asfr_df)

    # Add metadata columns
    single_year_df["source"] = "SDC_2024"
    single_year_df["notes"] = (
        "Interpolated from 5-year ASFRs using cubic spline; based on 2018-2022 ND births and population"
    )

    # Calculate TFR from raw data
    tfr = single_year_df["fertility_rate"].sum()
    print(f"\n   Total Fertility Rate (TFR) from raw data: {tfr:.4f}")
    print("   (This represents average children per woman over reproductive lifespan)")

    # Also calculate TFR from 5-year groups for comparison
    tfr_5yr = asfr_df["asfr"].sum() * 5  # Sum of annual rates × 5 years per group
    print(f"   TFR from 5-year groups: {tfr_5yr:.4f}")

    # Step 3b: Also create single-year rates from SDC blended rates
    print("\n6b. Interpolating SDC blended rates to single-year ages...")
    sdc_blended_df = interpolate_sdc_blended_rates(sdc_rates_df)
    tfr_blended = sdc_blended_df["fertility_rate"].sum()
    print(f"   TFR from SDC blended rates: {tfr_blended:.4f}")

    # Step 4: Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 4a. Single-year rates from raw data
    output_path = OUTPUT_DIR / "fertility_rates_sdc_2024.csv"
    single_year_df.to_csv(output_path, index=False)
    print("\n7. Saved single-year fertility rates (raw data) to:")
    print(f"   {output_path}")

    # 4b. 5-year group summary
    summary_path = OUTPUT_DIR / "fertility_rates_5yr_summary_sdc_2024.csv"
    summary_df = asfr_df.copy()
    summary_df["asfr_annual"] = summary_df["asfr"]
    summary_df["asfr_5yr_cumulative"] = summary_df["asfr"] * 5
    summary_df["source"] = "SDC_2024"
    summary_df["notes"] = "2018-2022 ND births / (avg 2018-2022 female pop × 5 years)"
    summary_df.to_csv(summary_path, index=False)
    print("\n8. Saved 5-year summary to:")
    print(f"   {summary_path}")

    # 4c. SDC blended rates (as used in their projections)
    blended_path = OUTPUT_DIR / "fertility_rates_sdc_blended_2024.csv"
    sdc_blended_df.to_csv(blended_path, index=False)
    print("\n9. Saved SDC blended rates to:")
    print(f"   {blended_path}")

    # Print final table
    print("\n" + "=" * 60)
    print("FINAL OUTPUT: Single-Year Fertility Rates (first 10 rows)")
    print("=" * 60)
    print(single_year_df.head(10).to_string(index=False))
    print("...")
    print(single_year_df.tail(5).to_string(index=False))

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Age range: {single_year_df['age'].min()} - {single_year_df['age'].max()}")
    print(
        f"Peak fertility age (raw): {single_year_df.loc[single_year_df['fertility_rate'].idxmax(), 'age']}"
    )
    print(f"Peak fertility rate (raw): {single_year_df['fertility_rate'].max():.6f}")
    print(f"Total Fertility Rate (TFR) from raw data: {tfr:.4f} children per woman")
    print(f"Total Fertility Rate (TFR) from SDC blended: {tfr_blended:.4f} children per woman")
    print("Sex ratio at birth (per SDC): 51.2% male, 48.8% female")

    print("\n" + "=" * 60)
    print("METHODOLOGY NOTES")
    print("=" * 60)
    print("""
The SDC blended rates differ from raw calculation because SDC:
1. Blends ND rates with national CDC NVSS 2021 rates
2. Applies smoothing to reduce county-level anomalies
3. Uses a wider time window (2016-2022 vs 2018-2022 for births)

For projections matching SDC methodology, use: fertility_rates_sdc_blended_2024.csv
For pure ND data-based rates, use: fertility_rates_sdc_2024.csv
""")

    return single_year_df


if __name__ == "__main__":
    main()
