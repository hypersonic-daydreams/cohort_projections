"""
Unit tests for PEP migration rate processing (process_pep_migration_rates).

Tests the BEBR-based multi-period averaging pipeline that converts Census PEP
county-level net migration data into age/sex/race-specific migration tables
using multi-period averaging and Rogers-Castro age distribution.

Uses synthetic data fixtures -- does not depend on actual PEP data files.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Standard demographic categories used throughout the project
RACE_CATEGORIES = [
    "White alone, Non-Hispanic",
    "Black alone, Non-Hispanic",
    "AIAN alone, Non-Hispanic",
    "Asian/PI alone, Non-Hispanic",
    "Two or more races, Non-Hispanic",
    "Hispanic (any race)",
]
SEX_CATEGORIES = ["Male", "Female"]
AGES = list(range(91))
COHORTS_PER_COUNTY = len(AGES) * len(SEX_CATEGORIES) * len(RACE_CATEGORIES)  # 1,092

# Test county GEOIDs (5-digit state+county FIPS)
OIL_COUNTY = "38105"  # Williams County (oil-producing)
METRO_COUNTY = "38017"  # Cass County (Fargo metro)
RURAL_COUNTY = "38073"  # Ransom County (rural)
TEST_COUNTIES = [OIL_COUNTY, METRO_COUNTY, RURAL_COUNTY]

_PEP_MODULE = "cohort_projections.data.process.pep_regime_analysis"

from cohort_projections.data.process.migration_rates import (
    process_pep_migration_rates,
)


@pytest.fixture
def mock_config():
    """Mock projection configuration matching production config structure."""
    return {
        "demographics": {
            "race_ethnicity": {
                "categories": RACE_CATEGORIES,
            },
            "sex": SEX_CATEGORIES,
            "age_groups": {"min_age": 0, "max_age": 90},
        },
        "output": {
            "compression": "gzip",
        },
        "rates": {
            "migration": {
                "domestic": {
                    "averaging_method": "BEBR_multiperiod",
                    "base_periods": {
                        "short": [2019, 2024],
                        "medium": [2014, 2024],
                        "long": [2005, 2024],
                        "full": [2000, 2024],
                    },
                    "dampening": {
                        "enabled": False,  # Disabled in tests to not affect existing assertions
                    },
                },
            },
        },
    }


@pytest.fixture
def synthetic_pep_data():
    """Create synthetic PEP county components data for 3 test counties.

    Mimics the structure produced by Phase 1 extraction:
    - 3 counties: oil (Williams), metro (Cass), rural (Ransom)
    - Years 2000-2024 with is_preferred_estimate=True
    - Realistic net migration patterns by regime
    """
    np.random.seed(42)
    records = []

    county_patterns = {
        OIL_COUNTY: {
            (2000, 2010): -200,
            (2011, 2015): 5000,
            (2016, 2021): -1500,
            (2022, 2024): 800,
        },
        METRO_COUNTY: {
            (2000, 2010): 500,
            (2011, 2015): 2000,
            (2016, 2021): 300,
            (2022, 2024): 1000,
        },
        RURAL_COUNTY: {
            (2000, 2010): -80,
            (2011, 2015): -30,
            (2016, 2021): -100,
            (2022, 2024): -50,
        },
    }

    for geoid, patterns in county_patterns.items():
        for (start_year, end_year), base_netmig in patterns.items():
            for year in range(start_year, end_year + 1):
                noise = np.random.normal(0, abs(base_netmig) * 0.1 + 10)
                records.append(
                    {
                        "geoid": geoid,
                        "state_fips": "38",
                        "county_fips": geoid[2:],
                        "year": year,
                        "netmig": base_netmig + noise,
                        "intl_mig": base_netmig * 0.1 + np.random.normal(0, 5),
                        "domestic_mig": base_netmig * 0.9 + np.random.normal(0, 10),
                        "state_name": "North Dakota",
                        "county_name": {
                            OIL_COUNTY: "Williams County",
                            METRO_COUNTY: "Cass County",
                            RURAL_COUNTY: "Ransom County",
                        }[geoid],
                        "vintage": f"{start_year}-{end_year}",
                        "is_preferred_estimate": True,
                    }
                )

    return pd.DataFrame(records)


@pytest.fixture
def synthetic_population():
    """Create base population data for 3 test counties.

    Provides all 1,092 cohort combinations per county with realistic
    population distributions.
    """
    np.random.seed(123)
    records = []

    county_pops = {
        OIL_COUNTY: 37000,
        METRO_COUNTY: 185000,
        RURAL_COUNTY: 5200,
    }

    race_proportions = {
        "White alone, Non-Hispanic": 0.80,
        "Black alone, Non-Hispanic": 0.03,
        "AIAN alone, Non-Hispanic": 0.05,
        "Asian/PI alone, Non-Hispanic": 0.02,
        "Two or more races, Non-Hispanic": 0.03,
        "Hispanic (any race)": 0.07,
    }

    for geoid, total_pop in county_pops.items():
        for age in AGES:
            if age < 5:
                age_weight = 0.06
            elif age < 18:
                age_weight = 0.05
            elif age < 30:
                age_weight = 0.07
            elif age < 50:
                age_weight = 0.06
            elif age < 65:
                age_weight = 0.05
            elif age < 80:
                age_weight = 0.03
            else:
                age_weight = 0.01

            for sex in SEX_CATEGORIES:
                sex_weight = 0.50
                for race, race_prop in race_proportions.items():
                    pop = max(
                        1,
                        int(
                            total_pop
                            * age_weight
                            / 91
                            * sex_weight
                            * race_prop
                            * (1 + np.random.uniform(-0.1, 0.1))
                        ),
                    )
                    records.append(
                        {
                            "county_fips": geoid,
                            "age": age,
                            "sex": sex,
                            "race_ethnicity": race,
                            "population": pop,
                        }
                    )

    return pd.DataFrame(records)


def _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population):
    """Write synthetic data to parquet files and return paths."""
    pop_file = tmp_path / "population.parquet"
    synthetic_population.to_parquet(pop_file, index=False)

    pep_file = tmp_path / "pep_data.parquet"
    synthetic_pep_data.to_parquet(pep_file, index=False)

    return pep_file, pop_file


class TestProcessPepMigrationRates:
    """Tests for the main process_pep_migration_rates function."""

    def test_output_correct_row_count(
        self,
        tmp_path,
        synthetic_pep_data,
        synthetic_population,
        mock_config,
    ):
        """Output has correct total number of rows (n_counties * 1,092)."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with patch(
            f"{_PEP_MODULE}.load_pep_preferred_estimates",
            side_effect=lambda p: synthetic_pep_data,
        ):
            results = process_pep_migration_rates(
                pep_path=pep_file,
                population_path=pop_file,
                output_dir=tmp_path / "output",
                config=mock_config,
                scenarios=["baseline"],
            )

        baseline_df = results["baseline"]
        expected_rows = len(TEST_COUNTIES) * COHORTS_PER_COUNTY
        assert len(baseline_df) == expected_rows

    def test_each_county_has_1092_rows(
        self,
        tmp_path,
        synthetic_pep_data,
        synthetic_population,
        mock_config,
    ):
        """Each county has exactly 1,092 rows (91 ages x 2 sexes x 6 races)."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with patch(
            f"{_PEP_MODULE}.load_pep_preferred_estimates",
            side_effect=lambda p: synthetic_pep_data,
        ):
            results = process_pep_migration_rates(
                pep_path=pep_file,
                population_path=pop_file,
                output_dir=tmp_path / "output",
                config=mock_config,
                scenarios=["baseline"],
            )

        baseline_df = results["baseline"]
        county_counts = baseline_df.groupby("county_fips").size()

        for geoid in TEST_COUNTIES:
            assert county_counts[geoid] == COHORTS_PER_COUNTY

    def test_required_columns_present(
        self,
        tmp_path,
        synthetic_pep_data,
        synthetic_population,
        mock_config,
    ):
        """Required columns are present in output."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with patch(
            f"{_PEP_MODULE}.load_pep_preferred_estimates",
            side_effect=lambda p: synthetic_pep_data,
        ):
            results = process_pep_migration_rates(
                pep_path=pep_file,
                population_path=pop_file,
                output_dir=tmp_path / "output",
                config=mock_config,
                scenarios=["baseline"],
            )

        baseline_df = results["baseline"]
        required_cols = [
            "county_fips",
            "age",
            "sex",
            "race_ethnicity",
            "net_migration",
        ]
        for col in required_cols:
            assert col in baseline_df.columns, f"Missing column: {col}"

    def test_age_distribution_peaks_around_25(
        self,
        tmp_path,
        synthetic_pep_data,
        synthetic_population,
        mock_config,
    ):
        """Age distribution peaks around age 25 (Rogers-Castro pattern)."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with patch(
            f"{_PEP_MODULE}.load_pep_preferred_estimates",
            side_effect=lambda p: synthetic_pep_data,
        ):
            results = process_pep_migration_rates(
                pep_path=pep_file,
                population_path=pop_file,
                output_dir=tmp_path / "output",
                config=mock_config,
                scenarios=["baseline"],
            )

        baseline_df = results["baseline"]
        metro = baseline_df[baseline_df["county_fips"] == METRO_COUNTY]

        # Get absolute migration by age
        age_mig = metro.groupby("age")["net_migration"].sum().abs()

        # Young adult ages (20-30) should have higher migration than elderly (70+)
        young_adult_avg = age_mig[(age_mig.index >= 20) & (age_mig.index <= 30)].mean()
        elderly_avg = age_mig[age_mig.index >= 70].mean()

        assert young_adult_avg > elderly_avg

    def test_both_sexes_present_equally(
        self,
        tmp_path,
        synthetic_pep_data,
        synthetic_population,
        mock_config,
    ):
        """Both sexes present in roughly equal proportions."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with patch(
            f"{_PEP_MODULE}.load_pep_preferred_estimates",
            side_effect=lambda p: synthetic_pep_data,
        ):
            results = process_pep_migration_rates(
                pep_path=pep_file,
                population_path=pop_file,
                output_dir=tmp_path / "output",
                config=mock_config,
                scenarios=["baseline"],
            )

        baseline_df = results["baseline"]
        sex_counts = baseline_df.groupby("sex").size()

        assert set(sex_counts.index) == {"Male", "Female"}
        assert sex_counts["Male"] == sex_counts["Female"]

    def test_all_six_race_categories_present(
        self,
        tmp_path,
        synthetic_pep_data,
        synthetic_population,
        mock_config,
    ):
        """All 6 race/ethnicity categories are present in the output."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with patch(
            f"{_PEP_MODULE}.load_pep_preferred_estimates",
            side_effect=lambda p: synthetic_pep_data,
        ):
            results = process_pep_migration_rates(
                pep_path=pep_file,
                population_path=pop_file,
                output_dir=tmp_path / "output",
                config=mock_config,
                scenarios=["baseline"],
            )

        baseline_df = results["baseline"]
        actual_races = set(baseline_df["race_ethnicity"].unique())
        expected_races = set(RACE_CATEGORIES)

        assert actual_races == expected_races

    def test_output_files_created(
        self,
        tmp_path,
        synthetic_pep_data,
        synthetic_population,
        mock_config,
    ):
        """Output files (parquet and CSV) are created for each scenario."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)
        output_dir = tmp_path / "output"

        with patch(
            f"{_PEP_MODULE}.load_pep_preferred_estimates",
            side_effect=lambda p: synthetic_pep_data,
        ):
            process_pep_migration_rates(
                pep_path=pep_file,
                population_path=pop_file,
                output_dir=output_dir,
                config=mock_config,
                scenarios=["baseline"],
            )

        assert (output_dir / "migration_rates_pep_baseline.parquet").exists()
        assert (output_dir / "migration_rates_pep_baseline.csv").exists()


class TestScenarioGeneration:
    """Tests for multi-scenario generation."""

    @pytest.fixture
    def _run_all_scenarios(
        self,
        tmp_path,
        synthetic_pep_data,
        synthetic_population,
        mock_config,
    ):
        """Run process_pep_migration_rates with all three scenarios."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with patch(
            f"{_PEP_MODULE}.load_pep_preferred_estimates",
            side_effect=lambda p: synthetic_pep_data,
        ):
            results = process_pep_migration_rates(
                pep_path=pep_file,
                population_path=pop_file,
                output_dir=tmp_path / "output",
                config=mock_config,
                scenarios=["baseline", "low", "high"],
            )

        return results

    def test_baseline_scenario_generated_by_default(
        self,
        tmp_path,
        synthetic_pep_data,
        synthetic_population,
        mock_config,
    ):
        """Baseline scenario is generated by default when no scenarios specified."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with patch(
            f"{_PEP_MODULE}.load_pep_preferred_estimates",
            side_effect=lambda p: synthetic_pep_data,
        ):
            # No scenarios argument -- should default to ["baseline"]
            results = process_pep_migration_rates(
                pep_path=pep_file,
                population_path=pop_file,
                output_dir=tmp_path / "output",
                config=mock_config,
            )

        assert "baseline" in results
        assert len(results) == 1

    def test_multiple_scenarios_produce_different_results(self, _run_all_scenarios):
        """Multiple scenarios produce different total migration results."""
        results = _run_all_scenarios

        assert len(results) == 3
        assert "baseline" in results
        assert "low" in results
        assert "high" in results

        baseline_total = results["baseline"]["net_migration"].sum()
        low_total = results["low"]["net_migration"].sum()
        high_total = results["high"]["net_migration"].sum()

        # They should not all be identical
        assert not (baseline_total == low_total == high_total)

    def test_low_scenario_lower_than_baseline(self, _run_all_scenarios):
        """Low scenario has lower total migration than baseline.

        The low scenario uses the minimum period mean (most pessimistic),
        so its algebraic sum should be lower than the trimmed baseline.
        """
        results = _run_all_scenarios

        baseline_total = results["baseline"]["net_migration"].sum()
        low_total = results["low"]["net_migration"].sum()

        # Low should have a lower algebraic sum (more negative / less positive)
        assert low_total < baseline_total

    def test_each_scenario_has_output_files(
        self,
        tmp_path,
        synthetic_pep_data,
        synthetic_population,
        mock_config,
    ):
        """Each scenario produces parquet and CSV output files."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)
        output_dir = tmp_path / "output"

        with patch(
            f"{_PEP_MODULE}.load_pep_preferred_estimates",
            side_effect=lambda p: synthetic_pep_data,
        ):
            process_pep_migration_rates(
                pep_path=pep_file,
                population_path=pop_file,
                output_dir=output_dir,
                config=mock_config,
                scenarios=["baseline", "low", "high"],
            )

        for scenario in ["baseline", "low", "high"]:
            assert (output_dir / f"migration_rates_pep_{scenario}.parquet").exists()
            assert (output_dir / f"migration_rates_pep_{scenario}.csv").exists()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_county_with_zero_net_migration(
        self,
        tmp_path,
        mock_config,
    ):
        """County with zero net migration produces all-zero migration rates."""
        zero_county = "38099"

        pep_records = [
            {
                "geoid": zero_county,
                "state_fips": "38",
                "county_fips": "099",
                "year": year,
                "netmig": 0.0,
                "intl_mig": 0.0,
                "domestic_mig": 0.0,
                "state_name": "North Dakota",
                "county_name": "Zero County",
                "vintage": "test",
                "is_preferred_estimate": True,
            }
            for year in range(2000, 2025)
        ]
        zero_pep = pd.DataFrame(pep_records)

        pop_records = [
            {
                "county_fips": zero_county,
                "age": age,
                "sex": sex,
                "race_ethnicity": race,
                "population": 10,
            }
            for age in AGES
            for sex in SEX_CATEGORIES
            for race in RACE_CATEGORIES
        ]
        zero_pop = pd.DataFrame(pop_records)

        pep_file = tmp_path / "pep_data.parquet"
        zero_pep.to_parquet(pep_file, index=False)
        pop_file = tmp_path / "population.parquet"
        zero_pop.to_parquet(pop_file, index=False)

        with patch(
            f"{_PEP_MODULE}.load_pep_preferred_estimates",
            side_effect=lambda p: zero_pep,
        ):
            results = process_pep_migration_rates(
                pep_path=pep_file,
                population_path=pop_file,
                output_dir=tmp_path / "output",
                config=mock_config,
                scenarios=["baseline"],
            )

        baseline_df = results["baseline"]
        assert len(baseline_df) == COHORTS_PER_COUNTY
        assert (baseline_df["net_migration"] == 0).all()

    def test_county_with_very_small_population(
        self,
        tmp_path,
        mock_config,
    ):
        """County with very small population still produces valid output."""
        small_county = "38097"

        pep_records = [
            {
                "geoid": small_county,
                "state_fips": "38",
                "county_fips": "097",
                "year": year,
                "netmig": -5.0,
                "intl_mig": 0.0,
                "domestic_mig": -5.0,
                "state_name": "North Dakota",
                "county_name": "Small County",
                "vintage": "test",
                "is_preferred_estimate": True,
            }
            for year in range(2000, 2025)
        ]
        small_pep = pd.DataFrame(pep_records)

        pop_records = [
            {
                "county_fips": small_county,
                "age": age,
                "sex": sex,
                "race_ethnicity": race,
                "population": 1,
            }
            for age in AGES
            for sex in SEX_CATEGORIES
            for race in RACE_CATEGORIES
        ]
        small_pop = pd.DataFrame(pop_records)

        pep_file = tmp_path / "pep_data.parquet"
        small_pep.to_parquet(pep_file, index=False)
        pop_file = tmp_path / "population.parquet"
        small_pop.to_parquet(pop_file, index=False)

        with patch(
            f"{_PEP_MODULE}.load_pep_preferred_estimates",
            side_effect=lambda p: small_pep,
        ):
            results = process_pep_migration_rates(
                pep_path=pep_file,
                population_path=pop_file,
                output_dir=tmp_path / "output",
                config=mock_config,
                scenarios=["baseline"],
            )

        baseline_df = results["baseline"]
        assert len(baseline_df) == COHORTS_PER_COUNTY
        assert baseline_df["county_fips"].iloc[0] == small_county
        total = baseline_df["net_migration"].sum()
        assert abs(total - (-5.0)) < 1.0

    def test_single_county_works(
        self,
        tmp_path,
        mock_config,
    ):
        """Processing a single county works correctly (does not require all 53)."""
        single_county = "38017"

        pep_records = [
            {
                "geoid": single_county,
                "state_fips": "38",
                "county_fips": "017",
                "year": year,
                "netmig": 500.0,
                "intl_mig": 50.0,
                "domestic_mig": 450.0,
                "state_name": "North Dakota",
                "county_name": "Cass County",
                "vintage": "test",
                "is_preferred_estimate": True,
            }
            for year in range(2000, 2025)
        ]
        single_pep = pd.DataFrame(pep_records)

        pop_records = [
            {
                "county_fips": single_county,
                "age": age,
                "sex": sex,
                "race_ethnicity": race,
                "population": 100,
            }
            for age in AGES
            for sex in SEX_CATEGORIES
            for race in RACE_CATEGORIES
        ]
        single_pop = pd.DataFrame(pop_records)

        pep_file = tmp_path / "pep_data.parquet"
        single_pep.to_parquet(pep_file, index=False)
        pop_file = tmp_path / "population.parquet"
        single_pop.to_parquet(pop_file, index=False)

        with patch(
            f"{_PEP_MODULE}.load_pep_preferred_estimates",
            side_effect=lambda p: single_pep,
        ):
            results = process_pep_migration_rates(
                pep_path=pep_file,
                population_path=pop_file,
                output_dir=tmp_path / "output",
                config=mock_config,
                scenarios=["baseline"],
            )

        baseline_df = results["baseline"]
        assert len(baseline_df) == COHORTS_PER_COUNTY
        assert baseline_df["county_fips"].nunique() == 1
        assert baseline_df["county_fips"].iloc[0] == single_county

    def test_population_file_not_found_raises_error(
        self,
        tmp_path,
        synthetic_pep_data,
        mock_config,
    ):
        """FileNotFoundError raised when population file does not exist."""
        pep_file = tmp_path / "pep_data.parquet"
        synthetic_pep_data.to_parquet(pep_file, index=False)
        nonexistent_pop = tmp_path / "nonexistent.parquet"

        with (
            patch(
                f"{_PEP_MODULE}.load_pep_preferred_estimates",
                side_effect=lambda p: synthetic_pep_data,
            ),
            pytest.raises(FileNotFoundError),
        ):
            process_pep_migration_rates(
                pep_path=pep_file,
                population_path=nonexistent_pop,
                output_dir=tmp_path / "output",
                config=mock_config,
            )

    def test_processing_date_column_present(
        self,
        tmp_path,
        synthetic_pep_data,
        synthetic_population,
        mock_config,
    ):
        """processing_date column is present in output."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with patch(
            f"{_PEP_MODULE}.load_pep_preferred_estimates",
            side_effect=lambda p: synthetic_pep_data,
        ):
            results = process_pep_migration_rates(
                pep_path=pep_file,
                population_path=pop_file,
                output_dir=tmp_path / "output",
                config=mock_config,
                scenarios=["baseline"],
            )

        baseline_df = results["baseline"]
        assert "processing_date" in baseline_df.columns
        assert baseline_df["processing_date"].nunique() == 1
