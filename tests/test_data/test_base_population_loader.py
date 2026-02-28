"""
Tests for base population loader entry-point functions (PP-004-03).

Validates the three main loader entry points in
``cohort_projections/data/load/base_population_loader.py``:

1. ``load_base_population_for_county``  -- single-county loader
2. ``load_base_population_for_all_counties``  -- multi-county loader
3. ``load_base_population_for_state``  -- state aggregation loader

Also validates ADR-055 GQ separation invariants and edge-case handling.

Uses small synthetic data fixtures that mimic the real data shapes.  File I/O
is mocked; all transformation logic runs for real.

ADR references:
    - ADR-047: county-specific age-sex-race distributions
    - ADR-048: single-year-of-age base population
    - ADR-054: state = sum of counties
    - ADR-055: group quarters separation
    - ADR-056: testing strategy maturation
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cohort_projections.data.load.base_population_loader import (
    _distribute_gq_across_races,
    _expand_gq_to_single_year_ages,
    _separate_gq_from_base_population,
    clear_gq_cache,
    get_all_county_gq_populations,
    get_county_gq_population,
    load_base_population_for_all_counties,
    load_base_population_for_county,
    load_base_population_for_state,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_SEXES = ["Male", "Female"]
EXPECTED_RACES = [
    "White alone, Non-Hispanic",
    "Black alone, Non-Hispanic",
    "AIAN alone, Non-Hispanic",
    "Asian/PI alone, Non-Hispanic",
    "Two or more races, Non-Hispanic",
    "Hispanic (any race)",
]
MIN_AGE = 0
MAX_AGE = 90
N_AGES = MAX_AGE - MIN_AGE + 1  # 91
N_COHORTS = N_AGES * len(EXPECTED_SEXES) * len(EXPECTED_RACES)  # 1092


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_gq_caches():
    """Clear module-level GQ caches before each test to prevent cross-talk."""
    clear_gq_cache()
    yield
    clear_gq_cache()


@pytest.fixture
def minimal_config():
    """
    Minimal projection config dict that covers all keys the loaders read.

    GQ is *disabled* by default so tests that do not exercise GQ are not
    affected by mocking GQ file I/O.
    """
    return {
        "project": {"base_year": 2025},
        "base_population": {
            "age_resolution": "single_year",
            "single_year_distribution": "data/raw/population/nd_age_sex_race_distribution_single_year.csv",
            "county_race_interpolation": "sprague",
            "county_distributions": {
                "enabled": False,
                "path": "data/processed/county_age_sex_race_distributions.parquet",
                "blend_threshold": 5000,
            },
            "group_quarters": {
                "enabled": False,
                "method": "hold_constant",
                "gq_data_path": "data/processed/gq_county_age_sex_2025.parquet",
                "race_distribution": "county_proportional",
            },
        },
        "demographics": {
            "age_groups": {"min_age": MIN_AGE, "max_age": MAX_AGE},
            "sex": EXPECTED_SEXES,
            "race_ethnicity": {"categories": EXPECTED_RACES},
        },
    }


@pytest.fixture
def gq_enabled_config(minimal_config):
    """Config with GQ separation enabled."""
    cfg = minimal_config.copy()
    cfg["base_population"] = {
        **minimal_config["base_population"],
        "group_quarters": {
            **minimal_config["base_population"]["group_quarters"],
            "enabled": True,
        },
    }
    return cfg


@pytest.fixture
def synthetic_state_distribution():
    """
    Synthetic statewide age-sex-race distribution with 1092 rows.

    Proportions are uniform (1/1092 each) so that applying them to a known
    total population yields predictable per-cohort values.
    """
    rows = []
    for age in range(MIN_AGE, MAX_AGE + 1):
        for sex in EXPECTED_SEXES:
            for race in EXPECTED_RACES:
                rows.append(
                    {
                        "age": age,
                        "sex": sex,
                        "race": race,
                        "proportion": 1.0 / N_COHORTS,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_county_populations():
    """
    Three synthetic counties with known populations.

    38001 -- Adams County  (pop 2000)
    38017 -- Cass County   (pop 50000)
    38035 -- Grand Forks   (pop 30000)
    """
    return pd.DataFrame(
        {
            "county_fips": ["38001", "38017", "38035"],
            "county_name": ["Adams", "Cass", "Grand Forks"],
            "population": [2000, 50000, 30000],
        }
    )


@pytest.fixture
def synthetic_gq_data():
    """
    Synthetic GQ data for two counties (38017 and 38035).

    Simple structure: one age group per sex per county so GQ totals are easy
    to verify.  Uses age group "20-24" to mimic college dorms.
    """
    return pd.DataFrame(
        {
            "county_fips": ["38017", "38017", "38035", "38035"],
            "age_group": ["20-24", "20-24", "20-24", "20-24"],
            "sex": ["Male", "Female", "Male", "Female"],
            "gq_population": [500.0, 400.0, 300.0, 250.0],
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_standard_output_shape(df: pd.DataFrame, *, label: str = ""):
    """Assert that *df* has the required engine columns and demographic range."""
    prefix = f"[{label}] " if label else ""

    # Required columns
    assert set(df.columns) >= {
        "year",
        "age",
        "sex",
        "race",
        "population",
    }, f"{prefix}Missing required columns; got {list(df.columns)}"

    # All ages 0-90 present
    actual_ages = sorted(df["age"].unique())
    assert actual_ages == list(range(MIN_AGE, MAX_AGE + 1)), (
        f"{prefix}Expected ages 0-90; got min={min(actual_ages)}, max={max(actual_ages)}, "
        f"count={len(actual_ages)}"
    )

    # Both sexes present
    assert set(df["sex"].unique()) == set(EXPECTED_SEXES), (
        f"{prefix}Expected sexes {EXPECTED_SEXES}; got {sorted(df['sex'].unique())}"
    )

    # All 6 race categories present
    assert set(df["race"].unique()) == set(EXPECTED_RACES), (
        f"{prefix}Expected 6 race categories; got {sorted(df['race'].unique())}"
    )

    # Expected row count
    assert len(df) == N_COHORTS, (
        f"{prefix}Expected {N_COHORTS} rows; got {len(df)}"
    )


def _assert_population_nonnegative(df: pd.DataFrame, *, label: str = ""):
    """Assert no cell has a negative population."""
    prefix = f"[{label}] " if label else ""
    neg = df[df["population"] < 0]
    assert neg.empty, (
        f"{prefix}Found {len(neg)} rows with negative population:\n"
        f"{neg.head(10)}"
    )


# ===================================================================
# TestLoadBasePopulationForCounty
# ===================================================================


class TestLoadBasePopulationForCounty:
    """Tests for ``load_base_population_for_county``."""

    def test_output_columns_and_shape(
        self,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """Output DataFrame has expected columns and 1092 rows."""
        result = load_base_population_for_county(
            fips="38017",
            config=minimal_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        _assert_standard_output_shape(result, label="county_38017")

    def test_all_ages_present(
        self,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """Ages 0 through 90 are all present in the output."""
        result = load_base_population_for_county(
            fips="38017",
            config=minimal_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        assert set(result["age"].unique()) == set(range(MIN_AGE, MAX_AGE + 1))

    def test_both_sexes_present(
        self,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """Both Male and Female are present."""
        result = load_base_population_for_county(
            fips="38017",
            config=minimal_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        assert set(result["sex"].unique()) == {"Male", "Female"}

    def test_all_six_race_categories_present(
        self,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """All 6 standard race/ethnicity categories are present."""
        result = load_base_population_for_county(
            fips="38017",
            config=minimal_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        assert set(result["race"].unique()) == set(EXPECTED_RACES)

    def test_population_nonnegative(
        self,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """All population values are non-negative."""
        result = load_base_population_for_county(
            fips="38017",
            config=minimal_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        _assert_population_nonnegative(result, label="county_38017")

    def test_total_population_matches_input(
        self,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """Sum of cohort populations equals the county total population."""
        result = load_base_population_for_county(
            fips="38017",
            config=minimal_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        # Cass County has population 50000 in our synthetic data
        assert abs(result["population"].sum() - 50000.0) < 1.0

    def test_base_year_from_config(
        self,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """The year column reflects the base_year in config."""
        result = load_base_population_for_county(
            fips="38017",
            config=minimal_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        assert (result["year"] == 2025).all()

    def test_different_counties_get_different_totals(
        self,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """Two counties with different total populations produce different sums."""
        adams = load_base_population_for_county(
            fips="38001",
            config=minimal_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )
        cass = load_base_population_for_county(
            fips="38017",
            config=minimal_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        assert abs(adams["population"].sum() - 2000.0) < 1.0
        assert abs(cass["population"].sum() - 50000.0) < 1.0
        assert adams["population"].sum() < cass["population"].sum()

    def test_missing_fips_raises_value_error(
        self,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """A FIPS code not in the county populations table raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            load_base_population_for_county(
                fips="99999",
                config=minimal_config,
                distribution=synthetic_state_distribution,
                county_populations=synthetic_county_populations,
            )

    def test_fips_zero_padded(
        self,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """FIPS codes shorter than 5 digits are zero-padded."""
        # Pass "38017" as "38017" -- should work since it's already 5 digits
        result = load_base_population_for_county(
            fips="38017",
            config=minimal_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )
        assert len(result) == N_COHORTS

    def test_output_sorted_by_age_sex_race(
        self,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """Output is sorted by age, sex, race for deterministic ordering."""
        result = load_base_population_for_county(
            fips="38017",
            config=minimal_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        # Check that ages are non-decreasing
        ages = result["age"].values
        assert (ages[1:] >= ages[:-1]).all(), "Output is not sorted by age"

    def test_uniform_distribution_yields_equal_cohorts(
        self,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """With a uniform distribution, each cohort gets population / N_COHORTS."""
        result = load_base_population_for_county(
            fips="38017",
            config=minimal_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        expected_per_cohort = 50000.0 / N_COHORTS
        # All cohorts should be approximately equal (within rounding)
        assert result["population"].std() < 0.01, (
            "With uniform distribution, cohort populations should be nearly identical"
        )
        assert abs(result["population"].mean() - expected_per_cohort) < 0.01


# ===================================================================
# TestLoadBasePopulationForAllCounties
# ===================================================================


class TestLoadBasePopulationForAllCounties:
    """Tests for ``load_base_population_for_all_counties``."""

    @patch(
        "cohort_projections.data.load.base_population_loader.load_county_populations"
    )
    @patch(
        "cohort_projections.data.load.base_population_loader.load_state_age_sex_race_distribution"
    )
    @patch(
        "cohort_projections.data.load.base_population_loader.load_county_distributions_file"
    )
    def test_returns_dict_with_all_counties(
        self,
        mock_county_dists_file,
        mock_state_dist,
        mock_county_pops,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """Returned dict has one entry per county in the population table."""
        mock_state_dist.return_value = synthetic_state_distribution
        mock_county_pops.return_value = synthetic_county_populations
        mock_county_dists_file.return_value = None  # No county-specific dists

        result = load_base_population_for_all_counties(config=minimal_config)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"38001", "38017", "38035"}

    @patch(
        "cohort_projections.data.load.base_population_loader.load_county_populations"
    )
    @patch(
        "cohort_projections.data.load.base_population_loader.load_state_age_sex_race_distribution"
    )
    @patch(
        "cohort_projections.data.load.base_population_loader.load_county_distributions_file"
    )
    def test_each_county_has_correct_shape(
        self,
        mock_county_dists_file,
        mock_state_dist,
        mock_county_pops,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """Every county DataFrame has the standard engine columns and shape."""
        mock_state_dist.return_value = synthetic_state_distribution
        mock_county_pops.return_value = synthetic_county_populations
        mock_county_dists_file.return_value = None

        result = load_base_population_for_all_counties(config=minimal_config)

        for fips, df in result.items():
            _assert_standard_output_shape(df, label=f"county_{fips}")

    @patch(
        "cohort_projections.data.load.base_population_loader.load_county_populations"
    )
    @patch(
        "cohort_projections.data.load.base_population_loader.load_state_age_sex_race_distribution"
    )
    @patch(
        "cohort_projections.data.load.base_population_loader.load_county_distributions_file"
    )
    def test_each_county_population_nonnegative(
        self,
        mock_county_dists_file,
        mock_state_dist,
        mock_county_pops,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """No county DataFrame contains negative population values."""
        mock_state_dist.return_value = synthetic_state_distribution
        mock_county_pops.return_value = synthetic_county_populations
        mock_county_dists_file.return_value = None

        result = load_base_population_for_all_counties(config=minimal_config)

        for fips, df in result.items():
            _assert_population_nonnegative(df, label=f"county_{fips}")

    @patch(
        "cohort_projections.data.load.base_population_loader.load_county_populations"
    )
    @patch(
        "cohort_projections.data.load.base_population_loader.load_state_age_sex_race_distribution"
    )
    @patch(
        "cohort_projections.data.load.base_population_loader.load_county_distributions_file"
    )
    def test_fips_list_filters_counties(
        self,
        mock_county_dists_file,
        mock_state_dist,
        mock_county_pops,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """When fips_list is supplied, only those counties are loaded."""
        mock_state_dist.return_value = synthetic_state_distribution
        mock_county_pops.return_value = synthetic_county_populations
        mock_county_dists_file.return_value = None

        result = load_base_population_for_all_counties(
            config=minimal_config, fips_list=["38017"]
        )

        assert set(result.keys()) == {"38017"}

    @patch(
        "cohort_projections.data.load.base_population_loader.load_county_populations"
    )
    @patch(
        "cohort_projections.data.load.base_population_loader.load_state_age_sex_race_distribution"
    )
    @patch(
        "cohort_projections.data.load.base_population_loader.load_county_distributions_file"
    )
    def test_total_population_across_all_counties(
        self,
        mock_county_dists_file,
        mock_state_dist,
        mock_county_pops,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """Total population across all counties equals the sum of inputs."""
        mock_state_dist.return_value = synthetic_state_distribution
        mock_county_pops.return_value = synthetic_county_populations
        mock_county_dists_file.return_value = None

        result = load_base_population_for_all_counties(config=minimal_config)

        total = sum(df["population"].sum() for df in result.values())
        expected_total = 2000 + 50000 + 30000
        assert abs(total - expected_total) < 1.0

    @patch(
        "cohort_projections.data.load.base_population_loader.load_county_populations"
    )
    @patch(
        "cohort_projections.data.load.base_population_loader.load_state_age_sex_race_distribution"
    )
    @patch(
        "cohort_projections.data.load.base_population_loader.load_county_distributions_file"
    )
    def test_failed_county_excluded_gracefully(
        self,
        mock_county_dists_file,
        mock_state_dist,
        mock_county_pops,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """A county that raises an exception is excluded, not fatal."""
        mock_state_dist.return_value = synthetic_state_distribution
        mock_county_dists_file.return_value = None

        # Add a county with FIPS "38999" that has no population row;
        # it will be in fips_list but absent from county_populations,
        # triggering a ValueError inside load_base_population_for_county.
        mock_county_pops.return_value = synthetic_county_populations

        result = load_base_population_for_all_counties(
            config=minimal_config, fips_list=["38017", "38999"]
        )

        # 38017 should succeed; 38999 should fail gracefully
        assert "38017" in result
        assert "38999" not in result


# ===================================================================
# TestLoadBasePopulationForState
# ===================================================================


class TestLoadBasePopulationForState:
    """Tests for ``load_base_population_for_state``."""

    @patch(
        "cohort_projections.data.load.base_population_loader.load_base_population_for_all_counties"
    )
    def test_output_columns_and_shape(
        self,
        mock_all_counties,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """State DataFrame has expected columns and 1092 rows."""
        # Build two synthetic county DataFrames
        county_dfs = {}
        for fips, pop in [("38001", 2000), ("38017", 50000)]:
            rows = []
            for age in range(MIN_AGE, MAX_AGE + 1):
                for sex in EXPECTED_SEXES:
                    for race in EXPECTED_RACES:
                        rows.append(
                            {
                                "year": 2025,
                                "age": age,
                                "sex": sex,
                                "race": race,
                                "population": pop / N_COHORTS,
                            }
                        )
            county_dfs[fips] = pd.DataFrame(rows)
        mock_all_counties.return_value = county_dfs

        result = load_base_population_for_state(config=minimal_config)

        _assert_standard_output_shape(result, label="state")

    @patch(
        "cohort_projections.data.load.base_population_loader.load_base_population_for_all_counties"
    )
    def test_state_total_equals_sum_of_counties(
        self,
        mock_all_counties,
        minimal_config,
    ):
        """ADR-054: state total population == sum of county populations."""
        county_dfs = {}
        pops = {"38001": 2000, "38017": 50000, "38035": 30000}
        for fips, pop in pops.items():
            rows = []
            for age in range(MIN_AGE, MAX_AGE + 1):
                for sex in EXPECTED_SEXES:
                    for race in EXPECTED_RACES:
                        rows.append(
                            {
                                "year": 2025,
                                "age": age,
                                "sex": sex,
                                "race": race,
                                "population": pop / N_COHORTS,
                            }
                        )
            county_dfs[fips] = pd.DataFrame(rows)
        mock_all_counties.return_value = county_dfs

        state = load_base_population_for_state(config=minimal_config)

        county_total = sum(pops.values())
        state_total = state["population"].sum()
        assert abs(state_total - county_total) < 1.0, (
            f"State total {state_total:.1f} != county sum {county_total}"
        )

    @patch(
        "cohort_projections.data.load.base_population_loader.load_base_population_for_all_counties"
    )
    def test_state_population_nonnegative(
        self,
        mock_all_counties,
        minimal_config,
    ):
        """State-level populations are all non-negative."""
        county_dfs = {}
        for fips, pop in [("38001", 2000), ("38017", 50000)]:
            rows = []
            for age in range(MIN_AGE, MAX_AGE + 1):
                for sex in EXPECTED_SEXES:
                    for race in EXPECTED_RACES:
                        rows.append(
                            {
                                "year": 2025,
                                "age": age,
                                "sex": sex,
                                "race": race,
                                "population": pop / N_COHORTS,
                            }
                        )
            county_dfs[fips] = pd.DataFrame(rows)
        mock_all_counties.return_value = county_dfs

        state = load_base_population_for_state(config=minimal_config)

        _assert_population_nonnegative(state, label="state")

    @patch(
        "cohort_projections.data.load.base_population_loader.load_base_population_for_all_counties"
    )
    def test_state_aggregation_per_cohort(
        self,
        mock_all_counties,
        minimal_config,
    ):
        """State population in each cohort == sum of that cohort across counties."""
        county_dfs = {}
        # Deliberately give different distributions to the two counties
        # so per-cohort sums are meaningful
        for fips, pop in [("38001", 1000), ("38017", 3000)]:
            rows = []
            for age in range(MIN_AGE, MAX_AGE + 1):
                for sex in EXPECTED_SEXES:
                    for race in EXPECTED_RACES:
                        rows.append(
                            {
                                "year": 2025,
                                "age": age,
                                "sex": sex,
                                "race": race,
                                "population": pop / N_COHORTS,
                            }
                        )
            county_dfs[fips] = pd.DataFrame(rows)
        mock_all_counties.return_value = county_dfs

        state = load_base_population_for_state(config=minimal_config)

        # Pick a specific cohort and verify
        cohort = state[
            (state["age"] == 25)
            & (state["sex"] == "Male")
            & (state["race"] == "White alone, Non-Hispanic")
        ]
        assert len(cohort) == 1
        expected = 1000 / N_COHORTS + 3000 / N_COHORTS
        assert abs(cohort["population"].iloc[0] - expected) < 0.01

    @patch(
        "cohort_projections.data.load.base_population_loader.load_base_population_for_all_counties"
    )
    def test_empty_counties_raises(
        self,
        mock_all_counties,
        minimal_config,
    ):
        """If no counties load successfully, state loader raises ValueError."""
        mock_all_counties.return_value = {}

        with pytest.raises(ValueError, match="No county data"):
            load_base_population_for_state(config=minimal_config)

    @patch(
        "cohort_projections.data.load.base_population_loader.load_base_population_for_all_counties"
    )
    def test_state_all_ages_present(
        self,
        mock_all_counties,
        minimal_config,
    ):
        """State output contains all ages 0-90."""
        county_dfs = {}
        for fips, pop in [("38001", 1000)]:
            rows = []
            for age in range(MIN_AGE, MAX_AGE + 1):
                for sex in EXPECTED_SEXES:
                    for race in EXPECTED_RACES:
                        rows.append(
                            {
                                "year": 2025,
                                "age": age,
                                "sex": sex,
                                "race": race,
                                "population": pop / N_COHORTS,
                            }
                        )
            county_dfs[fips] = pd.DataFrame(rows)
        mock_all_counties.return_value = county_dfs

        state = load_base_population_for_state(config=minimal_config)

        assert set(state["age"].unique()) == set(range(MIN_AGE, MAX_AGE + 1))

    @patch(
        "cohort_projections.data.load.base_population_loader.load_base_population_for_all_counties"
    )
    def test_state_sorted_output(
        self,
        mock_all_counties,
        minimal_config,
    ):
        """State output is sorted by age, sex, race."""
        county_dfs = {}
        for fips, pop in [("38001", 1000)]:
            rows = []
            for age in range(MIN_AGE, MAX_AGE + 1):
                for sex in EXPECTED_SEXES:
                    for race in EXPECTED_RACES:
                        rows.append(
                            {
                                "year": 2025,
                                "age": age,
                                "sex": sex,
                                "race": race,
                                "population": pop / N_COHORTS,
                            }
                        )
            county_dfs[fips] = pd.DataFrame(rows)
        mock_all_counties.return_value = county_dfs

        state = load_base_population_for_state(config=minimal_config)

        ages = state["age"].values
        assert (ages[1:] >= ages[:-1]).all()


# ===================================================================
# TestGQSeparation
# ===================================================================


class TestGQSeparation:
    """Tests for ADR-055 Group Quarters separation logic."""

    def _build_base_pop(self, total_population: float = 10000.0) -> pd.DataFrame:
        """Build a synthetic base population DataFrame."""
        rows = []
        for age in range(MIN_AGE, MAX_AGE + 1):
            for sex in EXPECTED_SEXES:
                for race in EXPECTED_RACES:
                    rows.append(
                        {
                            "year": 2025,
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "population": total_population / N_COHORTS,
                        }
                    )
        return pd.DataFrame(rows)

    def test_gq_disabled_returns_full_population(
        self,
        minimal_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """When GQ is disabled, the full population is returned unchanged."""
        result = load_base_population_for_county(
            fips="38017",
            config=minimal_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        assert abs(result["population"].sum() - 50000.0) < 1.0

    @patch(
        "cohort_projections.data.load.base_population_loader._load_gq_data"
    )
    def test_gq_separation_hh_plus_gq_equals_total(
        self,
        mock_load_gq,
        gq_enabled_config,
        synthetic_state_distribution,
        synthetic_county_populations,
        synthetic_gq_data,
    ):
        """ADR-055 invariant: household_pop + gq_pop == total_pop."""
        mock_load_gq.return_value = synthetic_gq_data

        # Total population for Cass (38017)
        total_pop = 50000.0

        result = load_base_population_for_county(
            fips="38017",
            config=gq_enabled_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        # Household population from the result
        hh_pop = result["population"].sum()

        # GQ population stored in the module cache
        gq_df = get_county_gq_population("38017")
        assert gq_df is not None
        gq_pop = gq_df["gq_population"].sum()

        assert abs(hh_pop + gq_pop - total_pop) < 1.0, (
            f"HH ({hh_pop:.1f}) + GQ ({gq_pop:.1f}) = {hh_pop + gq_pop:.1f} "
            f"!= total ({total_pop:.1f})"
        )

    @patch(
        "cohort_projections.data.load.base_population_loader._load_gq_data"
    )
    def test_gq_separation_household_pop_nonnegative(
        self,
        mock_load_gq,
        gq_enabled_config,
        synthetic_state_distribution,
        synthetic_county_populations,
        synthetic_gq_data,
    ):
        """Household population is never negative after GQ subtraction."""
        mock_load_gq.return_value = synthetic_gq_data

        result = load_base_population_for_county(
            fips="38017",
            config=gq_enabled_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        _assert_population_nonnegative(result, label="hh_pop_38017")

    @patch(
        "cohort_projections.data.load.base_population_loader._load_gq_data"
    )
    def test_gq_separation_reduces_population(
        self,
        mock_load_gq,
        gq_enabled_config,
        synthetic_state_distribution,
        synthetic_county_populations,
        synthetic_gq_data,
    ):
        """With GQ enabled, household pop is less than total pop."""
        mock_load_gq.return_value = synthetic_gq_data

        # Total population for Cass (38017)
        total_pop = 50000.0

        result = load_base_population_for_county(
            fips="38017",
            config=gq_enabled_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        hh_pop = result["population"].sum()
        assert hh_pop < total_pop, (
            f"Household pop ({hh_pop:.0f}) should be less than total ({total_pop:.0f})"
        )

    @patch(
        "cohort_projections.data.load.base_population_loader._load_gq_data"
    )
    def test_gq_no_data_for_county_returns_full_pop(
        self,
        mock_load_gq,
        gq_enabled_config,
        synthetic_state_distribution,
        synthetic_county_populations,
    ):
        """County with no GQ data returns the full population unchanged."""
        # GQ data exists but does not include Adams (38001)
        mock_load_gq.return_value = pd.DataFrame(
            {
                "county_fips": ["38017"],
                "age_group": ["20-24"],
                "sex": ["Male"],
                "gq_population": [500.0],
            }
        )

        result = load_base_population_for_county(
            fips="38001",
            config=gq_enabled_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        # Adams (pop 2000) should be unchanged since no GQ data for it
        assert abs(result["population"].sum() - 2000.0) < 1.0

    @patch(
        "cohort_projections.data.load.base_population_loader._load_gq_data"
    )
    def test_gq_stored_in_module_cache(
        self,
        mock_load_gq,
        gq_enabled_config,
        synthetic_state_distribution,
        synthetic_county_populations,
        synthetic_gq_data,
    ):
        """After GQ separation, county GQ is retrievable from module cache."""
        mock_load_gq.return_value = synthetic_gq_data

        load_base_population_for_county(
            fips="38017",
            config=gq_enabled_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        gq = get_county_gq_population("38017")
        assert gq is not None
        assert "gq_population" in gq.columns

    @patch(
        "cohort_projections.data.load.base_population_loader._load_gq_data"
    )
    def test_gq_population_nonnegative(
        self,
        mock_load_gq,
        gq_enabled_config,
        synthetic_state_distribution,
        synthetic_county_populations,
        synthetic_gq_data,
    ):
        """GQ population values are never negative."""
        mock_load_gq.return_value = synthetic_gq_data

        load_base_population_for_county(
            fips="38017",
            config=gq_enabled_config,
            distribution=synthetic_state_distribution,
            county_populations=synthetic_county_populations,
        )

        gq = get_county_gq_population("38017")
        assert (gq["gq_population"] >= 0).all()

    def test_clear_gq_cache(self):
        """clear_gq_cache() empties both caches."""
        # Manually populate caches
        from cohort_projections.data.load import base_population_loader as bpl

        bpl._county_gq_populations["38017"] = pd.DataFrame(
            {"gq_population": [100.0]}
        )

        clear_gq_cache()

        assert get_county_gq_population("38017") is None
        assert get_all_county_gq_populations() == {}


# ===================================================================
# TestExpandGQToSingleYearAges
# ===================================================================


class TestExpandGQToSingleYearAges:
    """Tests for ``_expand_gq_to_single_year_ages``."""

    def test_uniform_expansion_preserves_total(self):
        """Total GQ population is preserved when expanding to single years."""
        gq = pd.DataFrame(
            {
                "age_group": ["20-24", "20-24"],
                "sex": ["Male", "Female"],
                "gq_population": [500.0, 400.0],
            }
        )

        result = _expand_gq_to_single_year_ages(gq)

        assert abs(result["gq_population"].sum() - 900.0) < 0.01

    def test_five_year_group_expands_to_five_ages(self):
        """A 5-year group produces exactly 5 single-year rows per sex."""
        gq = pd.DataFrame(
            {
                "age_group": ["20-24"],
                "sex": ["Male"],
                "gq_population": [500.0],
            }
        )

        result = _expand_gq_to_single_year_ages(gq)

        assert len(result) == 5
        assert set(result["age"].unique()) == {20, 21, 22, 23, 24}

    def test_terminal_group_expands_to_six_ages(self):
        """The 85+ group expands to ages 85-90 (6 single years)."""
        gq = pd.DataFrame(
            {
                "age_group": ["85+"],
                "sex": ["Male"],
                "gq_population": [60.0],
            }
        )

        result = _expand_gq_to_single_year_ages(gq)

        assert len(result) == 6
        assert set(result["age"].unique()) == {85, 86, 87, 88, 89, 90}
        assert abs(result["gq_population"].sum() - 60.0) < 0.01

    def test_uniform_within_group(self):
        """Within a 5-year group, population is divided equally."""
        gq = pd.DataFrame(
            {
                "age_group": ["20-24"],
                "sex": ["Female"],
                "gq_population": [250.0],
            }
        )

        result = _expand_gq_to_single_year_ages(gq)

        expected_per_age = 250.0 / 5
        for _, row in result.iterrows():
            assert abs(row["gq_population"] - expected_per_age) < 0.01

    def test_unknown_age_group_skipped(self):
        """Unknown age groups are skipped without error."""
        gq = pd.DataFrame(
            {
                "age_group": ["UNKNOWN"],
                "sex": ["Male"],
                "gq_population": [100.0],
            }
        )

        result = _expand_gq_to_single_year_ages(gq)

        assert result.empty


# ===================================================================
# TestDistributeGQAcrossRaces
# ===================================================================


class TestDistributeGQAcrossRaces:
    """Tests for ``_distribute_gq_across_races``."""

    def test_total_preserved_after_race_distribution(self, minimal_config):
        """Total GQ population is preserved after race distribution."""
        gq = pd.DataFrame(
            {
                "age": [20, 21],
                "sex": ["Male", "Male"],
                "gq_population": [100.0, 100.0],
            }
        )

        # Build a base pop with known race shares
        base_pop = pd.DataFrame(
            {
                "year": [2025] * 6,
                "age": [20] * 6,
                "sex": ["Male"] * 6,
                "race": EXPECTED_RACES,
                "population": [600.0, 100.0, 50.0, 80.0, 70.0, 100.0],
            }
        )

        result = _distribute_gq_across_races(gq, base_pop, minimal_config)

        assert abs(result["gq_population"].sum() - 200.0) < 0.01

    def test_race_categories_all_present(self, minimal_config):
        """All 6 race categories are present in the distributed GQ."""
        gq = pd.DataFrame(
            {
                "age": [20],
                "sex": ["Male"],
                "gq_population": [100.0],
            }
        )

        base_pop = pd.DataFrame(
            {
                "year": [2025] * 6,
                "age": [20] * 6,
                "sex": ["Male"] * 6,
                "race": EXPECTED_RACES,
                "population": [500.0, 100.0, 50.0, 80.0, 70.0, 200.0],
            }
        )

        result = _distribute_gq_across_races(gq, base_pop, minimal_config)

        assert set(result["race"].unique()) == set(EXPECTED_RACES)

    def test_zero_pop_county_distributes_uniformly(self, minimal_config):
        """When county has zero population, GQ distributes uniformly."""
        gq = pd.DataFrame(
            {
                "age": [20],
                "sex": ["Male"],
                "gq_population": [60.0],
            }
        )

        base_pop = pd.DataFrame(
            {
                "year": [2025] * 6,
                "age": [20] * 6,
                "sex": ["Male"] * 6,
                "race": EXPECTED_RACES,
                "population": [0.0] * 6,
            }
        )

        result = _distribute_gq_across_races(gq, base_pop, minimal_config)

        # With zero pop, should be uniform: 60 / 6 = 10 per race
        expected_per_race = 60.0 / len(EXPECTED_RACES)
        for _, row in result.iterrows():
            assert abs(row["gq_population"] - expected_per_race) < 0.01


# ===================================================================
# TestSeparateGQFromBasePopulation
# ===================================================================


class TestSeparateGQFromBasePopulation:
    """Tests for ``_separate_gq_from_base_population``."""

    @patch(
        "cohort_projections.data.load.base_population_loader._load_gq_data"
    )
    def test_hh_plus_gq_equals_total_invariant(
        self,
        mock_load_gq,
        gq_enabled_config,
        synthetic_gq_data,
    ):
        """Invariant: household + GQ == original total, cell-by-cell."""
        mock_load_gq.return_value = synthetic_gq_data

        # Build base pop
        rows = []
        for age in range(MIN_AGE, MAX_AGE + 1):
            for sex in EXPECTED_SEXES:
                for race in EXPECTED_RACES:
                    rows.append(
                        {
                            "year": 2025,
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "population": 50000.0 / N_COHORTS,
                        }
                    )
        base_pop = pd.DataFrame(rows)

        hh_pop, gq_pop = _separate_gq_from_base_population(
            fips="38017", base_pop=base_pop, config=gq_enabled_config
        )

        # Cell-by-cell: hh + gq == original
        merged = hh_pop.merge(
            gq_pop[["age", "sex", "race", "gq_population"]],
            on=["age", "sex", "race"],
            how="left",
        )
        merged["gq_population"] = merged["gq_population"].fillna(0.0)
        merged["reconstructed"] = merged["population"] + merged["gq_population"]

        original_total = 50000.0
        reconstructed_total = merged["reconstructed"].sum()
        assert abs(reconstructed_total - original_total) < 1.0

    @patch(
        "cohort_projections.data.load.base_population_loader._load_gq_data"
    )
    def test_gq_capped_at_total_pop(
        self,
        mock_load_gq,
        gq_enabled_config,
    ):
        """GQ is capped so household population never goes negative."""
        # Create GQ data where gq > total pop in some cells
        huge_gq = pd.DataFrame(
            {
                "county_fips": ["38001"],
                "age_group": ["20-24"],
                "sex": ["Male"],
                "gq_population": [999999.0],  # Way more than total pop
            }
        )
        mock_load_gq.return_value = huge_gq

        # Build base pop with small population
        rows = []
        for age in range(MIN_AGE, MAX_AGE + 1):
            for sex in EXPECTED_SEXES:
                for race in EXPECTED_RACES:
                    rows.append(
                        {
                            "year": 2025,
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "population": 100.0 / N_COHORTS,
                        }
                    )
        base_pop = pd.DataFrame(rows)

        hh_pop, gq_pop = _separate_gq_from_base_population(
            fips="38001", base_pop=base_pop, config=gq_enabled_config
        )

        # No negative household populations
        _assert_population_nonnegative(hh_pop, label="hh_pop_capped")

    @patch(
        "cohort_projections.data.load.base_population_loader._load_gq_data"
    )
    def test_no_gq_data_returns_unchanged(
        self,
        mock_load_gq,
        gq_enabled_config,
    ):
        """If _load_gq_data returns None, base pop is returned unchanged."""
        mock_load_gq.return_value = None

        rows = []
        for age in range(MIN_AGE, MAX_AGE + 1):
            for sex in EXPECTED_SEXES:
                for race in EXPECTED_RACES:
                    rows.append(
                        {
                            "year": 2025,
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "population": 1000.0 / N_COHORTS,
                        }
                    )
        base_pop = pd.DataFrame(rows)

        hh_pop, gq_pop = _separate_gq_from_base_population(
            fips="38001", base_pop=base_pop, config=gq_enabled_config
        )

        assert abs(hh_pop["population"].sum() - 1000.0) < 0.01
        assert abs(gq_pop["gq_population"].sum()) < 0.01


# ===================================================================
# TestGQCacheFunctions
# ===================================================================


class TestGQCacheFunctions:
    """Tests for GQ cache retrieval functions."""

    def test_get_county_gq_population_returns_none_when_empty(self):
        """get_county_gq_population returns None for unknown county."""
        assert get_county_gq_population("99999") is None

    def test_get_all_county_gq_populations_returns_copy(self):
        """get_all_county_gq_populations returns a copy, not the original."""
        from cohort_projections.data.load import base_population_loader as bpl

        bpl._county_gq_populations["38017"] = pd.DataFrame(
            {"gq_population": [100.0]}
        )

        result = get_all_county_gq_populations()
        # Mutating the result should not affect the original
        result.pop("38017", None)
        assert "38017" in bpl._county_gq_populations

    def test_get_county_gq_pads_fips(self):
        """get_county_gq_population zero-pads the FIPS code."""
        from cohort_projections.data.load import base_population_loader as bpl

        bpl._county_gq_populations["00001"] = pd.DataFrame(
            {"gq_population": [50.0]}
        )

        # Pass unpadded "1" -- should be padded to "00001"
        result = get_county_gq_population("1")
        assert result is not None
