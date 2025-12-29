#!/usr/bin/env python3
"""
Process North Dakota immigrant profile data from multiple sources
into a consolidated parquet file for analysis.

Sources:
- Migration Policy Institute
- American Immigration Council
- Census Reporter
- Data USA
- USAFacts
"""

import json
from pathlib import Path

import pandas as pd

# Paths - Use project-level data directories
PROJECT_ROOT = Path(__file__).parent.parent.parent  # cohort_projections/

# Input: raw ND immigrant profile data
SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "immigration" / "nd_immigrant_profile"

# Output: analysis goes to project-level processed directory
ANALYSIS_DIR = PROJECT_ROOT / "data" / "processed" / "immigration" / "analysis"


def load_json_sources() -> dict:
    """Load all JSON source files."""
    sources = {}
    for json_file in SOURCE_DIR.glob("*.json"):
        with open(json_file) as f:
            sources[json_file.stem] = json.load(f)
    return sources


def create_time_series_df(sources: dict) -> pd.DataFrame:
    """Create time series of ND foreign-born population."""
    # Consolidated time series from multiple sources
    time_series_data = [
        {
            "year": 2010,
            "foreign_born_population": 16000,
            "foreign_born_percent": 2.5,
            "source": "ACS/Decennial Census",
        },
        {
            "year": 2013,
            "foreign_born_population": 21100,
            "foreign_born_percent": 2.9,
            "source": "ACS 1-year",
        },
        {
            "year": 2014,
            "foreign_born_population": 24000,
            "foreign_born_percent": 3.3,
            "source": "USAFacts/ACS",
        },
        {
            "year": 2018,
            "foreign_born_population": 29000,
            "foreign_born_percent": 3.9,
            "source": "ACS 5-year 2014-2018",
        },
        {
            "year": 2019,
            "foreign_born_population": 31000,
            "foreign_born_percent": 4.1,
            "source": "ACS 1-year",
        },
        {
            "year": 2022,
            "foreign_born_population": 36023,
            "foreign_born_percent": 4.9,
            "source": "ACS 1-year",
        },
        {
            "year": 2023,
            "foreign_born_population": 33300,
            "foreign_born_percent": 4.27,
            "source": "Data USA/ACS 1-year",
        },
        {
            "year": 2024,
            "foreign_born_population": 42000,
            "foreign_born_percent": 5.3,
            "source": "USAFacts estimate",
        },
    ]

    df = pd.DataFrame(time_series_data)
    df["state"] = "North Dakota"
    df["state_fips"] = "38"
    return df


def create_countries_of_origin_df(sources: dict) -> pd.DataFrame:
    """Create top countries of origin table."""
    countries_data = [
        {
            "country": "Philippines",
            "percent_of_immigrants": 8.0,
            "population_estimate": 2721,
            "margin_of_error": 1278,
            "source": "Data USA 2023",
        },
        {
            "country": "Bhutan",
            "percent_of_immigrants": 8.0,
            "population_estimate": None,
            "margin_of_error": None,
            "source": "AIC/MPI",
        },
        {
            "country": "Nepal",
            "percent_of_immigrants": 8.0,
            "population_estimate": None,
            "margin_of_error": None,
            "source": "AIC/MPI",
        },
        {
            "country": "Canada",
            "percent_of_immigrants": 6.0,
            "population_estimate": 2767,
            "margin_of_error": 1289,
            "source": "Data USA 2023",
        },
        {
            "country": "Liberia",
            "percent_of_immigrants": 6.0,
            "population_estimate": None,
            "margin_of_error": None,
            "source": "AIC/MPI",
        },
        {
            "country": "Mexico",
            "percent_of_immigrants": None,
            "population_estimate": 3032,
            "margin_of_error": 1349,
            "source": "Data USA 2023",
        },
    ]

    df = pd.DataFrame(countries_data)
    df["state"] = "North Dakota"
    df["state_fips"] = "38"
    df["data_year"] = 2023
    return df


def create_region_of_birth_df(sources: dict) -> pd.DataFrame:
    """Create region of birth breakdown."""
    # From Census Reporter data
    region_data = [
        {"region": "africa", "percent": 34.24, "population": 11920},
        {"region": "asia", "percent": 30.11, "population": 10483},
        {"region": "latin_america", "percent": 16.05, "population": 5588},
        {"region": "europe", "percent": 10.05, "population": 3500},
        {"region": "northern_america", "percent": 8.43, "population": 2934},
        {"region": "oceania", "percent": 1.11, "population": 388},
    ]

    df = pd.DataFrame(region_data)
    df["state"] = "North Dakota"
    df["state_fips"] = "38"
    df["data_year"] = 2023
    df["total_foreign_born"] = 31096
    return df


def create_characteristics_df(sources: dict) -> pd.DataFrame:
    """Create immigrant characteristics table."""
    # Education
    education_data = [
        {
            "category": "education",
            "subcategory": "high_school_or_less",
            "nd_percent": 41.0,
            "us_percent": None,
        },
        {
            "category": "education",
            "subcategory": "some_college_or_associates",
            "nd_percent": 24.0,
            "us_percent": None,
        },
        {
            "category": "education",
            "subcategory": "bachelors_or_higher",
            "nd_percent": 35.0,
            "us_percent": None,
        },
    ]

    # Citizenship
    citizenship_data = [
        {
            "category": "citizenship",
            "subcategory": "naturalized_citizen",
            "nd_percent": 45.0,
            "us_percent": None,
        },
        {
            "category": "citizenship",
            "subcategory": "undocumented",
            "nd_percent": 23.0,
            "us_percent": None,
        },
    ]

    # Labor force
    labor_data = [
        {
            "category": "labor_force",
            "subcategory": "percent_of_state_labor_force",
            "nd_percent": 6.0,
            "us_percent": None,
        },
        {
            "category": "labor_force",
            "subcategory": "percent_of_production_workers",
            "nd_percent": 13.0,
            "us_percent": None,
        },
        {
            "category": "labor_force",
            "subcategory": "percent_of_manufacturing_workers",
            "nd_percent": 11.0,
            "us_percent": None,
        },
        {
            "category": "labor_force",
            "subcategory": "brain_waste_underemployment",
            "nd_percent": 24.0,
            "us_percent": None,
        },
    ]

    # Population share
    population_data = [
        {
            "category": "population_share",
            "subcategory": "foreign_born_percent",
            "nd_percent": 4.0,
            "us_percent": 14.28,
        },
        {
            "category": "population_share",
            "subcategory": "native_born_with_immigrant_parent",
            "nd_percent": 5.0,
            "us_percent": None,
        },
    ]

    all_data = education_data + citizenship_data + labor_data + population_data
    df = pd.DataFrame(all_data)
    df["state"] = "North Dakota"
    df["state_fips"] = "38"
    df["data_year"] = 2023
    return df


def create_employment_by_industry_df(sources: dict) -> pd.DataFrame:
    """Create employment by industry table."""
    industry_data = [
        {"industry": "health_care_and_social_assistance", "immigrant_workers": 6245},
        {"industry": "educational_services", "immigrant_workers": 4690},
        {"industry": "manufacturing", "immigrant_workers": 3501},
        {"industry": "retail_trade", "immigrant_workers": 2156},
        {"industry": "wholesale_trade", "immigrant_workers": 2036},
    ]

    df = pd.DataFrame(industry_data)
    df["state"] = "North Dakota"
    df["state_fips"] = "38"
    df["data_year"] = 2023
    df["source"] = "American Immigration Council/Forum Together"
    return df


def create_diaspora_communities_df(sources: dict) -> pd.DataFrame:
    """Create growing diaspora communities table."""
    communities = [
        {"community": "Somali"},
        {"community": "Haitian"},
        {"community": "Liberian"},
        {"community": "Filipino"},
        {"community": "Indian"},
        {"community": "Rwandan"},
        {"community": "Sudanese"},
    ]

    df = pd.DataFrame(communities)
    df["state"] = "North Dakota"
    df["state_fips"] = "38"
    df["data_year"] = 2023
    df["source"] = "American Immigration Council stakeholder interviews"
    return df


def create_consolidated_parquet(sources: dict) -> None:
    """Create consolidated parquet file with all data tables."""
    # Create all dataframes
    time_series = create_time_series_df(sources)
    countries = create_countries_of_origin_df(sources)
    regions = create_region_of_birth_df(sources)
    characteristics = create_characteristics_df(sources)
    industries = create_employment_by_industry_df(sources)
    diaspora = create_diaspora_communities_df(sources)

    # Create summary statistics
    summary_data = {
        "metric": [
            "total_foreign_born_population",
            "foreign_born_percent_of_population",
            "us_foreign_born_percent",
            "nd_vs_us_ratio",
            "percent_change_2010_2022",
            "immigrant_business_owners",
            "immigrant_business_owners_percent",
            "international_migrants_2023",
            "international_migrants_2022",
            "open_positions_per_100_unemployed",
            "unemployment_rate_2023",
        ],
        "value": [
            31096.0,  # Total foreign born (Census Reporter)
            3.97,  # ND foreign born percent
            14.28,  # US foreign born percent
            0.28,  # ND is about 1/4 of US rate
            123.0,  # Percent change 2010-2022
            1056.0,  # Immigrant business owners
            3.0,  # Percent of self-employed
            7083.0,  # International migrants 2023
            4197.0,  # International migrants 2022
            100.0,  # Job openings per 100 unemployed
            1.9,  # Unemployment rate
        ],
        "source": [
            "Census Reporter ACS 2019-2023",
            "Census Reporter ACS 2019-2023",
            "Census Reporter ACS 2019-2023",
            "Calculated",
            "Migration Policy Institute",
            "American Immigration Council",
            "American Immigration Council",
            "American Immigration Council",
            "American Immigration Council",
            "ND Commerce OLI Report 2024",
            "ND Commerce OLI Report 2024",
        ],
    }
    summary = pd.DataFrame(summary_data)
    summary["state"] = "North Dakota"
    summary["state_fips"] = "38"
    summary["data_year"] = 2023

    # Combine all tables with a table_name column
    time_series["table_name"] = "time_series"
    countries["table_name"] = "countries_of_origin"
    regions["table_name"] = "region_of_birth"
    characteristics["table_name"] = "characteristics"
    industries["table_name"] = "employment_by_industry"
    diaspora["table_name"] = "diaspora_communities"
    summary["table_name"] = "summary_statistics"

    # Create output directory
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual tables as separate parquet files for easier querying
    output_path = ANALYSIS_DIR / "nd_immigrant_characteristics.parquet"

    # For the main consolidated file, we'll save the summary and characteristics
    # as they have compatible schemas
    main_df = pd.concat(
        [
            summary.assign(category="summary", subcategory=summary["metric"]).drop(
                columns=["metric"]
            ),
            characteristics.rename(columns={"nd_percent": "value", "us_percent": "us_comparison"}),
        ],
        ignore_index=True,
    )

    main_df.to_parquet(output_path, index=False)
    print(f"Saved consolidated data to: {output_path}")

    # Also save individual tables for convenience
    time_series.to_parquet(ANALYSIS_DIR / "nd_foreign_born_time_series.parquet", index=False)
    countries.to_parquet(ANALYSIS_DIR / "nd_countries_of_origin.parquet", index=False)
    regions.to_parquet(ANALYSIS_DIR / "nd_region_of_birth.parquet", index=False)
    industries.to_parquet(ANALYSIS_DIR / "nd_employment_by_industry.parquet", index=False)

    print(f"Saved time series data to: {ANALYSIS_DIR / 'nd_foreign_born_time_series.parquet'}")
    print(f"Saved countries of origin to: {ANALYSIS_DIR / 'nd_countries_of_origin.parquet'}")
    print(f"Saved region of birth to: {ANALYSIS_DIR / 'nd_region_of_birth.parquet'}")
    print(f"Saved employment by industry to: {ANALYSIS_DIR / 'nd_employment_by_industry.parquet'}")


def main():
    """Main entry point."""
    print("Loading source data...")
    sources = load_json_sources()
    print(f"Loaded {len(sources)} source files")

    print("\nProcessing data into parquet format...")
    create_consolidated_parquet(sources)

    print("\nDone!")


if __name__ == "__main__":
    main()
