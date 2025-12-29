# mypy: ignore-errors
"""
Process DHS Refugee and Naturalization Data

This script extracts data from:
1. DHS Refugee Flow Reports (2019-2023) - state and country tables
2. DHS Naturalization Yearbook data (2021-2023) - state and country tables

Output files:
- dhs_refugee_admissions.parquet - Refugee arrivals by state and country
- dhs_naturalizations_by_state.parquet - Naturalizations by state and country of birth
"""

from pathlib import Path

import pandas as pd

# Paths - Use project-level data directories
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # cohort_projections/

# Input: raw DHS data
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "immigration" / "dhs_refugees_naturalization"

# Output: analysis goes to project-level processed directory
ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"


def clean_column_name(name: str) -> str:
    """Convert column name to lowercase with underscores."""
    if pd.isna(name):
        return "unknown"
    name = str(name).strip().lower()
    name = name.replace(" ", "_").replace("/", "_").replace("-", "_")
    name = name.replace("__", "_")
    return name


def extract_refugee_state_data() -> pd.DataFrame:
    """
    Extract refugee arrivals by state from the 2023 report.
    Data from Table 5 in the PDF (manually transcribed since PDF tables need manual extraction).
    """
    # Data from Table 5 of the 2023 Refugee Flow Report
    # Refugee Arrivals by State of Residence: Fiscal Years 2021 to 2023
    state_data = [
        {"state": "Texas", "fy2021": 930, "fy2022": 2110, "fy2023": 5050},
        {"state": "New York", "fy2021": 710, "fy2022": 1400, "fy2023": 3850},
        {"state": "California", "fy2021": 990, "fy2022": 2180, "fy2023": 3670},
        {"state": "Pennsylvania", "fy2021": 400, "fy2022": 1100, "fy2023": 2770},
        {"state": "North Carolina", "fy2021": 500, "fy2022": 1150, "fy2023": 2610},
        {"state": "Arizona", "fy2021": 420, "fy2022": 1030, "fy2023": 2610},
        {"state": "Kentucky", "fy2021": 670, "fy2022": 1300, "fy2023": 2520},
        {"state": "Ohio", "fy2021": 450, "fy2022": 1130, "fy2023": 2500},
        {"state": "Michigan", "fy2021": 530, "fy2022": 1140, "fy2023": 2450},
        {"state": "Washington", "fy2021": 480, "fy2022": 1240, "fy2023": 2440},
        {"state": "Other", "fy2021": 5370, "fy2022": 11730, "fy2023": 29600},
        {"state": "Total", "fy2021": 11450, "fy2022": 25520, "fy2023": 60050},
    ]

    df = pd.DataFrame(state_data)

    # Melt to long format
    df_long = df.melt(
        id_vars=["state"],
        value_vars=["fy2021", "fy2022", "fy2023"],
        var_name="fiscal_year",
        value_name="refugee_arrivals",
    )
    df_long["fiscal_year"] = df_long["fiscal_year"].str.replace("fy", "").astype(int)

    return df_long


def extract_refugee_country_data() -> pd.DataFrame:
    """
    Extract refugee arrivals by country of nationality from the 2023 report.
    Data from Table 3 in the PDF.
    """
    # Data from Table 3 of the 2023 Refugee Flow Report
    # Refugee Arrivals by Country of Nationality: Fiscal Years 2021 to 2023
    country_data = [
        {"country": "Congo, Democratic Republic", "fy2021": 4880, "fy2022": 7740, "fy2023": 18080},
        {"country": "Syria", "fy2021": 1260, "fy2022": 4560, "fy2023": 10780},
        {"country": "Afghanistan", "fy2021": 870, "fy2022": 1620, "fy2023": 6590},
        {"country": "Burma", "fy2021": 770, "fy2022": 2140, "fy2023": 6130},
        {"country": "Guatemala", "fy2021": 60, "fy2022": 1080, "fy2023": 1760},
        {"country": "Sudan", "fy2021": 510, "fy2022": 1670, "fy2023": 1630},
        {"country": "Somalia", "fy2021": 200, "fy2022": 490, "fy2023": 1410},
        {"country": "Venezuela", "fy2021": 0, "fy2022": 160, "fy2023": 1370},
        {"country": "Ukraine", "fy2021": 800, "fy2022": 1590, "fy2023": 1340},
        {"country": "Iraq", "fy2021": 500, "fy2022": 500, "fy2023": 1220},
        {"country": "All other countries", "fy2021": 1610, "fy2022": 3970, "fy2023": 9750},
        {"country": "Total", "fy2021": 11450, "fy2022": 25520, "fy2023": 60050},
    ]

    df = pd.DataFrame(country_data)

    # Melt to long format
    df_long = df.melt(
        id_vars=["country"],
        value_vars=["fy2021", "fy2022", "fy2023"],
        var_name="fiscal_year",
        value_name="refugee_arrivals",
    )
    df_long["fiscal_year"] = df_long["fiscal_year"].str.replace("fy", "").astype(int)

    return df_long


def extract_refugee_region_data() -> pd.DataFrame:
    """
    Extract refugee arrivals by region from the 2023 report.
    Data from Table 1 in the PDF.
    """
    # Data from Table 1 - Proposed and Actual Refugee Admissions by Region
    region_data = [
        {
            "region": "Africa",
            "fy2021_ceiling": 22000,
            "fy2021_admissions": 6250,
            "fy2022_ceiling": 40000,
            "fy2022_admissions": 11390,
            "fy2023_ceiling": 40000,
            "fy2023_admissions": 24510,
        },
        {
            "region": "East Asia",
            "fy2021_ceiling": 6000,
            "fy2021_admissions": 780,
            "fy2022_ceiling": 15000,
            "fy2022_admissions": 2220,
            "fy2023_ceiling": 15000,
            "fy2023_admissions": 6260,
        },
        {
            "region": "Europe/Central Asia",
            "fy2021_ceiling": 4000,
            "fy2021_admissions": 980,
            "fy2022_ceiling": 10000,
            "fy2022_admissions": 2350,
            "fy2023_ceiling": 15000,
            "fy2023_admissions": 2770,
        },
        {
            "region": "Latin America/Caribbean",
            "fy2021_ceiling": 5000,
            "fy2021_admissions": 400,
            "fy2022_ceiling": 15000,
            "fy2022_admissions": 2490,
            "fy2023_ceiling": 15000,
            "fy2023_admissions": 6320,
        },
        {
            "region": "Near East/South Asia",
            "fy2021_ceiling": 13000,
            "fy2021_admissions": 3050,
            "fy2022_ceiling": 35000,
            "fy2022_admissions": 7080,
            "fy2023_ceiling": 35000,
            "fy2023_admissions": 20200,
        },
        {
            "region": "Total",
            "fy2021_ceiling": 62500,
            "fy2021_admissions": 11450,
            "fy2022_ceiling": 125000,
            "fy2022_admissions": 25520,
            "fy2023_ceiling": 125000,
            "fy2023_admissions": 60050,
        },
    ]

    return pd.DataFrame(region_data)


def process_naturalization_data() -> pd.DataFrame:
    """
    Extract naturalization data from DHS Yearbook Excel files.
    Focus on Table 22 (by state over time) and Supplemental Table 1 (by state and country).
    """
    all_data = []

    # Process FY2023 data
    try:
        df_2023 = pd.read_excel(
            SOURCE_DIR / "naturalizations_fy2023.xlsx", sheet_name="Table 22", header=5
        )
        df_2023 = df_2023.dropna(how="all")
        df_2023.columns = [clean_column_name(c) for c in df_2023.columns]

        # Rename first column to state
        cols = list(df_2023.columns)
        cols[0] = "state"
        df_2023.columns = cols

        # Keep only state and year columns
        year_cols = [c for c in df_2023.columns if c.startswith("20")]
        df_2023 = df_2023[["state"] + year_cols]

        # Melt to long format
        df_long = df_2023.melt(
            id_vars=["state"], var_name="fiscal_year", value_name="naturalizations"
        )
        df_long["fiscal_year"] = df_long["fiscal_year"].str.replace(".0", "").astype(int)
        all_data.append(df_long)
        print(
            f"Processed FY2023 naturalization data: {len(df_2023)} states, {len(year_cols)} years"
        )
    except Exception as e:
        print(f"Error processing FY2023: {e}")

    # Process FY2022 data
    try:
        df_2022 = pd.read_excel(
            SOURCE_DIR / "naturalizations_fy2022.xlsx", sheet_name="Table 22", header=5
        )
        df_2022 = df_2022.dropna(how="all")
        df_2022.columns = [clean_column_name(c) for c in df_2022.columns]

        # Rename first column to state
        cols = list(df_2022.columns)
        cols[0] = "state"
        df_2022.columns = cols

        # Keep only state and year columns
        year_cols = [c for c in df_2022.columns if c.startswith("20")]
        df_2022 = df_2022[["state"] + year_cols]

        # Melt to long format
        df_long = df_2022.melt(
            id_vars=["state"], var_name="fiscal_year", value_name="naturalizations"
        )
        df_long["fiscal_year"] = df_long["fiscal_year"].str.replace(".0", "").astype(int)
        # Only add years not already present
        print(
            f"Processed FY2022 naturalization data: {len(df_2022)} states, {len(year_cols)} years"
        )
    except Exception as e:
        print(f"Error processing FY2022: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        # Remove duplicates, keeping the most recent version of each state-year
        combined = combined.drop_duplicates(subset=["state", "fiscal_year"], keep="first")
        return combined
    return pd.DataFrame()


def process_naturalization_by_state_and_country() -> pd.DataFrame:
    """
    Extract naturalization data by state and country of birth from Supplemental Table 1.
    """
    try:
        df = pd.read_excel(
            SOURCE_DIR / "naturalizations_fy2023.xlsx", sheet_name="NATZSuppTable1", header=5
        )
        df = df.dropna(how="all")
        df.columns = [clean_column_name(c) for c in df.columns]

        # Rename first column
        cols = list(df.columns)
        cols[0] = "country_of_birth"
        df.columns = cols

        # Remove header rows
        df = df[~df["country_of_birth"].isin(["REGION", "COUNTRY", "region_and_country_of_birth"])]
        df = df[df["country_of_birth"].notna()]

        # Add fiscal year
        df["fiscal_year"] = 2023

        print(f"Processed naturalization by state and country: {len(df)} countries")
        return df
    except Exception as e:
        print(f"Error processing supplemental table: {e}")
        return pd.DataFrame()


def add_state_population_data() -> pd.DataFrame:
    """
    Add state population estimates for per-capita calculations.
    2022 Census estimates used as reference.
    """
    # State populations from Census Bureau 2022 estimates (NST-EST2022-POP)
    state_pops = {
        "Alabama": 5074296,
        "Alaska": 733583,
        "Arizona": 7359197,
        "Arkansas": 3045637,
        "California": 39029342,
        "Colorado": 5839926,
        "Connecticut": 3626205,
        "Delaware": 1018396,
        "District of Columbia": 671803,
        "Florida": 22244823,
        "Georgia": 10912876,
        "Guam": 172952,
        "Hawaii": 1440196,
        "Idaho": 1939033,
        "Illinois": 12582032,
        "Indiana": 6833037,
        "Iowa": 3200517,
        "Kansas": 2937150,
        "Kentucky": 4512310,
        "Louisiana": 4590241,
        "Maine": 1385340,
        "Maryland": 6164660,
        "Massachusetts": 6981974,
        "Michigan": 10034113,
        "Minnesota": 5717184,
        "Mississippi": 2940057,
        "Missouri": 6177957,
        "Montana": 1122867,
        "Nebraska": 1967923,
        "Nevada": 3177772,
        "New Hampshire": 1395231,
        "New Jersey": 9261699,
        "New Mexico": 2113344,
        "New York": 19677151,
        "North Carolina": 10698973,
        "North Dakota": 779261,
        "Ohio": 11756058,
        "Oklahoma": 4019800,
        "Oregon": 4240137,
        "Pennsylvania": 12972008,
        "Puerto Rico": 3221789,
        "Rhode Island": 1093734,
        "South Carolina": 5282634,
        "South Dakota": 909824,
        "Tennessee": 7051339,
        "Texas": 30029572,
        "Utah": 3380800,
        "Vermont": 647064,
        "Virginia": 8683619,
        "Washington": 7785786,
        "West Virginia": 1775156,
        "Wisconsin": 5892539,
        "Wyoming": 581381,
    }
    return pd.DataFrame([{"state": k, "population_2022": v} for k, v in state_pops.items()])


def main():
    print("=" * 60)
    print("Processing DHS Refugee and Naturalization Data")
    print("=" * 60)

    # Create analysis directory
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Process refugee data
    print("\n--- Refugee Admissions Data ---")

    refugee_by_state = extract_refugee_state_data()
    refugee_by_country = extract_refugee_country_data()
    refugee_by_region = extract_refugee_region_data()

    # Add state populations for per-capita
    state_pops = add_state_population_data()

    # Merge refugee data with populations
    refugee_by_state_with_pop = refugee_by_state.merge(state_pops, on="state", how="left")
    refugee_by_state_with_pop["refugees_per_100k"] = (
        refugee_by_state_with_pop["refugee_arrivals"]
        / refugee_by_state_with_pop["population_2022"]
        * 100000
    ).round(2)

    # Print summary
    print(f"\nRefugee by state: {len(refugee_by_state)} records")
    print(f"Refugee by country: {len(refugee_by_country)} records")
    print(f"Refugee by region: {len(refugee_by_region)} records")

    # Top states by 2023 per-capita rate
    top_states = (
        refugee_by_state_with_pop[
            (refugee_by_state_with_pop["fiscal_year"] == 2023)
            & (refugee_by_state_with_pop["state"] != "Total")
            & (refugee_by_state_with_pop["state"] != "Other")
        ]
        .sort_values("refugees_per_100k", ascending=False)
        .head(10)
    )

    print("\nTop 10 states by refugee resettlement per capita (FY2023):")
    for _, row in top_states.iterrows():
        print(
            f"  {row['state']}: {row['refugees_per_100k']:.1f} per 100k ({row['refugee_arrivals']:,} total)"
        )

    # Save refugee data
    refugee_output = ANALYSIS_DIR / "dhs_refugee_admissions.parquet"
    # Convert dict of dataframes to single dataframe with category
    refugee_combined = pd.concat(
        [
            refugee_by_state_with_pop.assign(category="by_state"),
            refugee_by_country.assign(category="by_country"),
        ],
        ignore_index=True,
    )
    refugee_combined.to_parquet(refugee_output, index=False)
    print(f"\nSaved refugee data to: {refugee_output}")

    # Process naturalization data
    print("\n--- Naturalization Data ---")

    naturalization_by_state = process_naturalization_data()
    naturalization_detail = process_naturalization_by_state_and_country()

    if not naturalization_by_state.empty:
        # Add populations
        naturalization_by_state = naturalization_by_state.merge(state_pops, on="state", how="left")
        naturalization_by_state["naturalizations_per_100k"] = (
            naturalization_by_state["naturalizations"]
            / naturalization_by_state["population_2022"]
            * 100000
        ).round(2)

        # Print top states
        top_nat_states = (
            naturalization_by_state[
                (naturalization_by_state["fiscal_year"] == 2023)
                & (naturalization_by_state["state"] != "Total")
                & (naturalization_by_state["naturalizations"].notna())
            ]
            .sort_values("naturalizations", ascending=False)
            .head(10)
        )

        print("\nTop 10 states by naturalizations (FY2023):")
        for _, row in top_nat_states.iterrows():
            print(f"  {row['state']}: {int(row['naturalizations']):,}")

        # Find North Dakota's position
        nd_data = naturalization_by_state[
            (naturalization_by_state["state"] == "North Dakota")
            & (naturalization_by_state["fiscal_year"] == 2023)
        ]
        if not nd_data.empty:
            nd_nat = nd_data.iloc[0]["naturalizations"]
            nd_rank = (
                naturalization_by_state[
                    (naturalization_by_state["fiscal_year"] == 2023)
                    & (naturalization_by_state["state"] != "Total")
                ]["naturalizations"]
                .rank(ascending=False)
                .loc[nd_data.index[0]]
            )
            print(f"\nNorth Dakota FY2023: {int(nd_nat):,} naturalizations (rank #{int(nd_rank)})")

        # Save naturalization data
        nat_output = ANALYSIS_DIR / "dhs_naturalizations_by_state.parquet"
        naturalization_by_state.to_parquet(nat_output, index=False)
        print(f"\nSaved naturalization data to: {nat_output}")

    # Save detailed state-country data if available
    if not naturalization_detail.empty:
        detail_output = ANALYSIS_DIR / "dhs_naturalizations_by_state_country.parquet"
        naturalization_detail.to_parquet(detail_output, index=False)
        print(f"Saved detailed naturalization data to: {detail_output}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
