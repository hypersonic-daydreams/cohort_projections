"""
End-to-End Integration Tests for Cohort Component Projection System.

These tests use actual processed data files to run real projections and
validate that the cohort component projection engine produces demographically
sensible results.

Test Focus: Cass County (FIPS 38017) - Fargo area, largest county in ND
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cohort_projections.core.cohort_component import CohortComponentProjection  # noqa: E402


class TestEndToEndProjection:
    """End-to-end integration tests using real processed data."""

    # Data file paths
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
    FERTILITY_FILE = DATA_DIR / "fertility" / "asfr_processed.csv"
    SURVIVAL_FILE = DATA_DIR / "mortality" / "survival_rates_processed.csv"
    MIGRATION_FILE = DATA_DIR / "migration" / "nd_migration_processed.csv"
    COUNTY_POP_FILE = DATA_DIR / "population" / "nd_county_population.csv"
    AGE_SEX_RACE_FILE = DATA_DIR / "population" / "nd_age_sex_race_distribution.csv"

    # Cass County FIPS code (Fargo area)
    CASS_COUNTY_FIPS = "38017"

    @pytest.fixture
    def processed_data(self):
        """Load all processed data files."""
        data = {}

        # Load fertility rates
        fertility_df = pd.read_csv(self.FERTILITY_FILE)
        data["fertility"] = fertility_df

        # Load survival rates
        survival_df = pd.read_csv(self.SURVIVAL_FILE)
        data["survival"] = survival_df

        # Load migration data
        migration_df = pd.read_csv(self.MIGRATION_FILE)
        data["migration"] = migration_df

        # Load county population totals
        county_pop_df = pd.read_csv(self.COUNTY_POP_FILE)
        data["county_population"] = county_pop_df

        # Load age-sex-race distribution
        age_sex_race_df = pd.read_csv(self.AGE_SEX_RACE_FILE)
        data["age_sex_race"] = age_sex_race_df

        return data

    @pytest.fixture
    def cass_county_population(self, processed_data):
        """Build base population for Cass County using actual data."""
        county_pop = processed_data["county_population"]
        age_sex_race = processed_data["age_sex_race"]

        # Get Cass County total population
        cass_row = county_pop[county_pop["county_fips"] == int(self.CASS_COUNTY_FIPS)]
        if cass_row.empty:
            pytest.skip("Cass County data not found in county population file")

        cass_total_pop = cass_row["population_2024"].values[0]

        # Calculate proportions from age-sex-race distribution
        total_proportion = age_sex_race["proportion"].sum()

        # Build base population by applying proportions to Cass County total
        base_pop_records = []
        for _, row in age_sex_race.iterrows():
            # Parse age group to get single-year ages
            age_group = row["age_group"]
            if age_group == "85+":
                # For 85+ we'll use age 85-90 range
                ages = list(range(85, 91))
            else:
                age_start, age_end = map(int, age_group.split("-"))
                ages = list(range(age_start, age_end + 1))

            # Distribute this proportion across the ages in the group
            population_in_group = cass_total_pop * (row["proportion"] / total_proportion)
            pop_per_age = population_in_group / len(ages)

            for age in ages:
                # Map race names to projection system format
                race_map = {
                    "white_nonhispanic": "White alone, Non-Hispanic",
                    "black_nonhispanic": "Black alone, Non-Hispanic",
                    "aian_nonhispanic": "AIAN alone, Non-Hispanic",
                    "asian_nonhispanic": "Asian/PI alone, Non-Hispanic",
                    "nhpi_nonhispanic": "Asian/PI alone, Non-Hispanic",  # Combine with Asian
                    "multiracial_nonhispanic": "Two or more races, Non-Hispanic",
                    "other_nonhispanic": "Two or more races, Non-Hispanic",  # Map other to multiracial
                    "hispanic": "Hispanic (any race)",
                }

                race_ethnicity = row["race_ethnicity"]
                mapped_race = race_map.get(race_ethnicity, "Two or more races, Non-Hispanic")

                base_pop_records.append(
                    {
                        "year": 2025,
                        "age": min(age, 90),  # Cap at 90 for 90+ group
                        "sex": row["sex"].title(),
                        "race": mapped_race,
                        "population": pop_per_age,
                    }
                )

        base_pop_df = pd.DataFrame(base_pop_records)

        # Aggregate by year/age/sex/race (in case of duplicates from mapping)
        base_pop_df = base_pop_df.groupby(["year", "age", "sex", "race"], as_index=False).agg(
            {"population": "sum"}
        )

        return base_pop_df

    @pytest.fixture
    def fertility_rates(self, processed_data):
        """Prepare fertility rates for projection engine."""
        fertility_df = processed_data["fertility"]

        # Convert ASFR from per-1000 to per-woman (proportion)
        fertility_records = []

        # Map race categories
        race_map = {
            "total": "White alone, Non-Hispanic",  # Use total rates for all races as baseline
            "white_nh": "White alone, Non-Hispanic",
            "black_nh": "Black alone, Non-Hispanic",
            "aian_nh": "AIAN alone, Non-Hispanic",
            "asian_nh": "Asian/PI alone, Non-Hispanic",
            "hispanic": "Hispanic (any race)",
        }

        # Define all race categories for the projection
        all_races = [
            "White alone, Non-Hispanic",
            "Black alone, Non-Hispanic",
            "AIAN alone, Non-Hispanic",
            "Asian/PI alone, Non-Hispanic",
            "Two or more races, Non-Hispanic",
            "Hispanic (any race)",
        ]

        # Process fertility data
        for _, row in fertility_df.iterrows():
            age_group = row["age"]
            race = row["race_ethnicity"]
            asfr = row["asfr"] / 1000  # Convert from per-1000 to proportion

            if race not in race_map:
                continue

            # Parse age group to individual ages
            if isinstance(age_group, str) and "-" in age_group:
                age_start, age_end = map(int, age_group.split("-"))
                ages = list(range(age_start, age_end + 1))
            else:
                continue

            mapped_race = race_map[race]

            for age in ages:
                fertility_records.append(
                    {
                        "age": age,
                        "race": mapped_race,
                        "fertility_rate": asfr,
                    }
                )

        fertility_rate_df = pd.DataFrame(fertility_records)

        # Average if there are duplicates
        fertility_rate_df = fertility_rate_df.groupby(["age", "race"], as_index=False).agg(
            {"fertility_rate": "mean"}
        )

        # Add missing race categories using total rates
        total_rates = fertility_rate_df[
            fertility_rate_df["race"] == "White alone, Non-Hispanic"
        ].copy()

        for race in all_races:
            if race not in fertility_rate_df["race"].values:
                race_rates = total_rates.copy()
                race_rates["race"] = race
                fertility_rate_df = pd.concat([fertility_rate_df, race_rates], ignore_index=True)

        return fertility_rate_df

    @pytest.fixture
    def survival_rates(self, processed_data):
        """Prepare survival rates for projection engine."""
        survival_df = processed_data["survival"]

        # Map race and sex categories
        race_map = {
            "aian_nh": "AIAN alone, Non-Hispanic",
            "asian_nh": "Asian/PI alone, Non-Hispanic",
            "black_nh": "Black alone, Non-Hispanic",
            "hispanic": "Hispanic (any race)",
            "white_nh": "White alone, Non-Hispanic",
        }

        sex_map = {
            "female": "Female",
            "male": "Male",
        }

        survival_records = []
        for _, row in survival_df.iterrows():
            race = row.get("race_ethnicity", row.get("race"))
            sex = row.get("sex", "")

            if race not in race_map or sex not in sex_map:
                continue

            survival_records.append(
                {
                    "age": int(row["age"]),
                    "sex": sex_map[sex],
                    "race": race_map[race],
                    "survival_rate": float(row["survival_rate"]),
                }
            )

        survival_rate_df = pd.DataFrame(survival_records)

        # Add missing race categories - use white_nh as proxy
        all_races = [
            "White alone, Non-Hispanic",
            "Black alone, Non-Hispanic",
            "AIAN alone, Non-Hispanic",
            "Asian/PI alone, Non-Hispanic",
            "Two or more races, Non-Hispanic",
            "Hispanic (any race)",
        ]

        for sex in ["Male", "Female"]:
            for race in all_races:
                existing = survival_rate_df[
                    (survival_rate_df["sex"] == sex) & (survival_rate_df["race"] == race)
                ]
                if existing.empty:
                    # Use white_nh rates as proxy
                    proxy_rates = survival_rate_df[
                        (survival_rate_df["sex"] == sex)
                        & (survival_rate_df["race"] == "White alone, Non-Hispanic")
                    ].copy()
                    proxy_rates["race"] = race
                    survival_rate_df = pd.concat([survival_rate_df, proxy_rates], ignore_index=True)

        return survival_rate_df

    @pytest.fixture
    def migration_rates(self, processed_data, cass_county_population):
        """Prepare migration rates for projection engine."""
        migration_df = processed_data["migration"]
        base_pop = cass_county_population

        # Get Cass County migration data - use average of available years
        cass_migration = migration_df[migration_df["county_fips"] == int(self.CASS_COUNTY_FIPS)]

        if cass_migration.empty:
            # Return zero migration if no data
            migration_records = []
            for _, row in base_pop.iterrows():
                migration_records.append(
                    {
                        "age": row["age"],
                        "sex": row["sex"],
                        "race": row["race"],
                        "net_migration": 0.0,
                    }
                )
            return pd.DataFrame(migration_records)

        # Calculate average annual net migration
        avg_net_migration = cass_migration["net_migration"].mean()

        # Distribute migration proportionally across cohorts
        total_pop = base_pop["population"].sum()

        migration_records = []
        for _, row in base_pop.iterrows():
            # Migration follows age patterns: higher for young adults
            age = row["age"]
            if 18 <= age <= 34:
                age_factor = 1.5  # Higher migration for young adults
            elif 35 <= age <= 54:
                age_factor = 1.0
            elif age >= 65:
                age_factor = 0.8  # Lower for elderly
            else:
                age_factor = 0.5  # Lower for children

            pop_proportion = row["population"] / total_pop if total_pop > 0 else 0
            net_mig = avg_net_migration * pop_proportion * age_factor

            migration_records.append(
                {
                    "age": row["age"],
                    "sex": row["sex"],
                    "race": row["race"],
                    "net_migration": net_mig,
                }
            )

        migration_rate_df = pd.DataFrame(migration_records)

        # Normalize to match total expected migration
        current_total = migration_rate_df["net_migration"].sum()
        if current_total != 0:
            scale_factor = avg_net_migration / current_total
            migration_rate_df["net_migration"] = migration_rate_df["net_migration"] * scale_factor

        return migration_rate_df

    @pytest.fixture
    def projection_config(self):
        """Configuration for the 5-year test projection."""
        return {
            "project": {
                "name": "Integration Test - Cass County",
                "base_year": 2025,
                "projection_horizon": 5,
            },
            "demographics": {
                "age_groups": {
                    "min_age": 0,
                    "max_age": 90,
                },
            },
            "rates": {
                "fertility": {
                    "apply_to_ages": [15, 49],
                    "sex_ratio_male": 0.51,
                },
                "mortality": {
                    "improvement_factor": 0.0,  # No improvement for test
                },
            },
        }

    def test_data_files_exist(self):
        """Test that all required data files exist."""
        assert self.FERTILITY_FILE.exists(), f"Fertility file not found: {self.FERTILITY_FILE}"
        assert self.SURVIVAL_FILE.exists(), f"Survival file not found: {self.SURVIVAL_FILE}"
        assert self.MIGRATION_FILE.exists(), f"Migration file not found: {self.MIGRATION_FILE}"
        assert self.COUNTY_POP_FILE.exists(), (
            f"County population file not found: {self.COUNTY_POP_FILE}"
        )
        assert self.AGE_SEX_RACE_FILE.exists(), (
            f"Age-sex-race file not found: {self.AGE_SEX_RACE_FILE}"
        )

    def test_base_population_is_reasonable(self, cass_county_population):
        """Test that base population for Cass County is reasonable."""
        base_pop = cass_county_population
        total_pop = base_pop["population"].sum()

        # Cass County should have around 180-220K population
        assert 150_000 < total_pop < 250_000, (
            f"Cass County population {total_pop:,.0f} outside expected range (150K-250K)"
        )

        # Check for negative populations
        assert (base_pop["population"] >= 0).all(), "Base population contains negative values"

        # Check sex distribution (allow wider range for sample-based data)
        # Note: The distribution data is from a sample and may have some skew
        male_pop = base_pop[base_pop["sex"] == "Male"]["population"].sum()
        female_pop = base_pop[base_pop["sex"] == "Female"]["population"].sum()
        sex_ratio = male_pop / female_pop if female_pop > 0 else 0

        assert 0.8 < sex_ratio < 1.25, (
            f"Sex ratio {sex_ratio:.2f} outside expected range (0.8-1.25)"
        )

    def test_fertility_rates_are_valid(self, fertility_rates):
        """Test that fertility rates are valid."""
        # Check for required columns
        assert "age" in fertility_rates.columns
        assert "race" in fertility_rates.columns
        assert "fertility_rate" in fertility_rates.columns

        # Check rate values are reasonable (0 to 0.2 births per woman per year)
        assert (fertility_rates["fertility_rate"] >= 0).all(), "Negative fertility rates found"
        assert (fertility_rates["fertility_rate"] <= 0.2).all(), (
            "Implausibly high fertility rates found (> 0.2)"
        )

        # Peak fertility should be in ages 25-34
        peak_ages = fertility_rates[(fertility_rates["age"] >= 25) & (fertility_rates["age"] <= 34)]
        other_ages = fertility_rates[(fertility_rates["age"] < 25) | (fertility_rates["age"] > 34)]

        avg_peak = peak_ages["fertility_rate"].mean()
        avg_other = other_ages["fertility_rate"].mean()

        assert avg_peak > avg_other, "Peak fertility should be in ages 25-34"

    def test_survival_rates_are_valid(self, survival_rates):
        """Test that survival rates are valid."""
        # Check for required columns
        assert "age" in survival_rates.columns
        assert "sex" in survival_rates.columns
        assert "race" in survival_rates.columns
        assert "survival_rate" in survival_rates.columns

        # Check rate values are in valid range
        assert (survival_rates["survival_rate"] >= 0).all(), "Negative survival rates found"
        assert (survival_rates["survival_rate"] <= 1).all(), "Survival rates > 1 found"

        # Infant survival should be high (> 0.99)
        infant_rates = survival_rates[survival_rates["age"] == 0]
        if not infant_rates.empty:
            min_infant_survival = infant_rates["survival_rate"].min()
            assert min_infant_survival > 0.98, (
                f"Infant survival rate {min_infant_survival:.4f} is too low"
            )

        # Elderly survival should be lower than young adult survival
        young_adult = survival_rates[(survival_rates["age"] >= 20) & (survival_rates["age"] <= 40)]
        elderly = survival_rates[survival_rates["age"] >= 80]

        avg_young = young_adult["survival_rate"].mean()
        avg_elderly = elderly["survival_rate"].mean()

        assert avg_young > avg_elderly, "Young adult survival should be higher than elderly"

    def test_five_year_projection_runs(
        self,
        cass_county_population,
        fertility_rates,
        survival_rates,
        migration_rates,
        projection_config,
    ):
        """Test that a 5-year projection runs without errors."""
        projection = CohortComponentProjection(
            base_population=cass_county_population,
            fertility_rates=fertility_rates,
            survival_rates=survival_rates,
            migration_rates=migration_rates,
            config=projection_config,
        )

        results = projection.run_projection(
            start_year=2025,
            end_year=2030,
            scenario="baseline",
        )

        # Check results structure
        assert not results.empty, "Projection results are empty"
        assert "year" in results.columns
        assert "age" in results.columns
        assert "sex" in results.columns
        assert "race" in results.columns
        assert "population" in results.columns

        # Check we have data for all projection years
        years = results["year"].unique()
        expected_years = [2025, 2026, 2027, 2028, 2029, 2030]
        for year in expected_years:
            assert year in years, f"Missing year {year} in results"

    def test_population_changes_are_plausible(
        self,
        cass_county_population,
        fertility_rates,
        survival_rates,
        migration_rates,
        projection_config,
    ):
        """Test that year-over-year population changes are plausible."""
        projection = CohortComponentProjection(
            base_population=cass_county_population,
            fertility_rates=fertility_rates,
            survival_rates=survival_rates,
            migration_rates=migration_rates,
            config=projection_config,
        )

        results = projection.run_projection(
            start_year=2025,
            end_year=2030,
            scenario="baseline",
        )

        # Calculate total population by year
        pop_by_year = results.groupby("year")["population"].sum().sort_index()

        # Check year-over-year changes are within +/- 3%
        for i in range(1, len(pop_by_year)):
            prev_pop = pop_by_year.iloc[i - 1]
            curr_pop = pop_by_year.iloc[i]
            change_pct = (curr_pop - prev_pop) / prev_pop

            assert -0.03 <= change_pct <= 0.03, (
                f"Year {pop_by_year.index[i]}: Population change {change_pct:.1%} "
                f"exceeds +/- 3% threshold"
            )

    def test_no_negative_populations(
        self,
        cass_county_population,
        fertility_rates,
        survival_rates,
        migration_rates,
        projection_config,
    ):
        """Test that no cohort has negative population."""
        projection = CohortComponentProjection(
            base_population=cass_county_population,
            fertility_rates=fertility_rates,
            survival_rates=survival_rates,
            migration_rates=migration_rates,
            config=projection_config,
        )

        results = projection.run_projection(
            start_year=2025,
            end_year=2030,
            scenario="baseline",
        )

        negative_pops = results[results["population"] < 0]
        assert negative_pops.empty, f"Found {len(negative_pops)} cohorts with negative population"

    def test_births_are_reasonable(
        self,
        cass_county_population,
        fertility_rates,
        survival_rates,
        migration_rates,
        projection_config,
    ):
        """Test that annual births are in expected range."""
        projection = CohortComponentProjection(
            base_population=cass_county_population,
            fertility_rates=fertility_rates,
            survival_rates=survival_rates,
            migration_rates=migration_rates,
            config=projection_config,
        )

        results = projection.run_projection(
            start_year=2025,
            end_year=2030,
            scenario="baseline",
        )

        # Calculate females of reproductive age (15-44) in base year
        base_pop = cass_county_population
        females_15_44 = base_pop[
            (base_pop["sex"] == "Female") & (base_pop["age"] >= 15) & (base_pop["age"] <= 44)
        ]["population"].sum()

        # Expected births should be ~2-5% of females 15-44 per year
        # For Cass County ~200K, females 15-44 might be ~35-40K
        # So births should be ~700-2000 per year
        # Note: With sample-based age distribution, female count may be lower,
        # leading to higher apparent birth rates

        # Check births (age 0) in each year
        for year in range(2026, 2031):
            births = results[(results["year"] == year) & (results["age"] == 0)]["population"].sum()

            # Expected range: 0.015 to 0.06 times females 15-44
            # Wider range due to sample-based distribution data
            min_births = females_15_44 * 0.015
            max_births = females_15_44 * 0.06

            assert min_births < births < max_births, (
                f"Year {year}: Births {births:,.0f} outside expected range "
                f"({min_births:,.0f} - {max_births:,.0f})"
            )

    def test_age_structure_evolves_correctly(
        self,
        cass_county_population,
        fertility_rates,
        survival_rates,
        migration_rates,
        projection_config,
    ):
        """Test that age structure evolves as expected over time."""
        projection = CohortComponentProjection(
            base_population=cass_county_population,
            fertility_rates=fertility_rates,
            survival_rates=survival_rates,
            migration_rates=migration_rates,
            config=projection_config,
        )

        results = projection.run_projection(
            start_year=2025,
            end_year=2030,
            scenario="baseline",
        )

        # Get median age for start and end years
        def calculate_median_age(pop_df):
            ages = []
            for _, row in pop_df.iterrows():
                ages.extend([row["age"]] * int(row["population"]))
            return np.median(ages) if ages else 0

        start_pop = results[results["year"] == 2025]
        end_pop = results[results["year"] == 2030]

        start_median = calculate_median_age(start_pop)
        end_median = calculate_median_age(end_pop)

        # Median age should not change dramatically in 5 years
        # (typical change is < 1 year per 5 years of projection)
        age_change = abs(end_median - start_median)
        assert age_change < 3, (
            f"Median age changed by {age_change:.1f} years, "
            f"which is unusually high for a 5-year period"
        )

    def test_summary_statistics_available(
        self,
        cass_county_population,
        fertility_rates,
        survival_rates,
        migration_rates,
        projection_config,
    ):
        """Test that summary statistics are generated correctly."""
        projection = CohortComponentProjection(
            base_population=cass_county_population,
            fertility_rates=fertility_rates,
            survival_rates=survival_rates,
            migration_rates=migration_rates,
            config=projection_config,
        )

        projection.run_projection(
            start_year=2025,
            end_year=2030,
            scenario="baseline",
        )

        summary = projection.get_projection_summary()

        # Check summary has expected columns
        assert not summary.empty, "Summary is empty"
        assert "year" in summary.columns
        assert "total_population" in summary.columns

        # Check we have summaries for all projection years (except base)
        expected_years = [2026, 2027, 2028, 2029, 2030]
        summary_years = summary["year"].tolist()
        for year in expected_years:
            assert year in summary_years, f"Missing summary for year {year}"


class TestDataIntegrity:
    """Tests for data file integrity and format."""

    DATA_DIR = PROJECT_ROOT / "data" / "raw"

    def test_fertility_file_format(self):
        """Test fertility file has expected format."""
        fertility_df = pd.read_csv(self.DATA_DIR / "fertility" / "asfr_processed.csv")

        assert "age" in fertility_df.columns, "Missing 'age' column"
        assert "race_ethnicity" in fertility_df.columns or "race" in fertility_df.columns, (
            "Missing race column"
        )
        assert "asfr" in fertility_df.columns, "Missing 'asfr' column"

        # Check age groups cover reproductive ages
        ages = fertility_df["age"].unique()
        age_strings = [str(a) for a in ages]
        has_young = any("15" in a or "20" in a for a in age_strings)
        has_peak = any("25" in a or "30" in a for a in age_strings)

        assert has_young and has_peak, "Fertility data missing key age groups"

    def test_survival_file_format(self):
        """Test survival file has expected format."""
        survival_df = pd.read_csv(self.DATA_DIR / "mortality" / "survival_rates_processed.csv")

        assert "age" in survival_df.columns, "Missing 'age' column"
        assert "sex" in survival_df.columns, "Missing 'sex' column"
        assert "survival_rate" in survival_df.columns, "Missing 'survival_rate' column"

        # Check ages cover full range
        ages = survival_df["age"].unique()
        assert 0 in ages, "Missing age 0"
        assert max(ages) >= 85, "Missing elderly ages"

    def test_migration_file_format(self):
        """Test migration file has expected format."""
        migration_df = pd.read_csv(self.DATA_DIR / "migration" / "nd_migration_processed.csv")

        assert "county_fips" in migration_df.columns, "Missing 'county_fips' column"
        assert "net_migration" in migration_df.columns, "Missing 'net_migration' column"

        # Check we have data for Cass County
        cass_data = migration_df[migration_df["county_fips"] == 38017]
        assert not cass_data.empty, "Missing Cass County migration data"

    def test_county_population_file_format(self):
        """Test county population file has expected format."""
        county_df = pd.read_csv(self.DATA_DIR / "population" / "nd_county_population.csv")

        assert "county_fips" in county_df.columns, "Missing 'county_fips' column"
        assert "population_2024" in county_df.columns, "Missing 'population_2024' column"

        # Check we have all 53 ND counties
        n_counties = len(county_df)
        assert n_counties >= 50, f"Expected ~53 counties, found {n_counties}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
