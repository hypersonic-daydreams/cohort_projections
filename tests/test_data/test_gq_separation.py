"""
Tests for Group Quarters (GQ) separation functions (ADR-055).

Covers the five GQ-related functions with 0% test coverage:
1. subtract_gq_from_populations (residual_migration.py) -- Phase 2 core
2. _separate_gq_from_base_population (base_population_loader.py)
3. get_county_gq_population (base_population_loader.py)
4. _expand_gq_to_single_year_ages (base_population_loader.py)
5. _distribute_gq_across_races (base_population_loader.py)

Uses small synthetic DataFrames (not mocks) per ADR-056 testing guidelines.
Focuses on invariant assertions: conservation, non-negativity, graceful fallbacks.
"""

import pandas as pd
import pytest

from cohort_projections.data.load.base_population_loader import (
    AGE_GROUP_RANGES,
    _distribute_gq_across_races,
    _expand_gq_to_single_year_ages,
    _separate_gq_from_base_population,
    clear_gq_cache,
    get_county_gq_population,
    _county_gq_populations,
    _gq_data_cache,
)
from cohort_projections.data.process.residual_migration import (
    subtract_gq_from_populations,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

STANDARD_RACES = [
    "White alone, Non-Hispanic",
    "Black alone, Non-Hispanic",
    "AIAN alone, Non-Hispanic",
    "Asian/PI alone, Non-Hispanic",
    "Two or more races, Non-Hispanic",
    "Hispanic (any race)",
]

THREE_AGE_GROUPS = ["0-4", "20-24", "85+"]

FIVE_AGE_GROUPS = ["0-4", "5-9", "20-24", "65-69", "85+"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_gq_module_state():
    """Clear GQ module-level caches before and after each test."""
    clear_gq_cache()
    yield
    clear_gq_cache()


@pytest.fixture
def minimal_config():
    """Minimal config dict enabling GQ separation."""
    return {
        "demographics": {
            "sex": ["Male", "Female"],
            "race_ethnicity": {
                "categories": STANDARD_RACES,
            },
            "age_groups": {"min_age": 0, "max_age": 90},
        },
        "base_population": {
            "group_quarters": {
                "enabled": True,
                "gq_data_path": "data/processed/gq_county_age_sex_2025.parquet",
            },
        },
    }


@pytest.fixture
def gq_county_3ages():
    """Synthetic GQ data for one county, 3 age groups x 2 sexes."""
    records = []
    for ag in THREE_AGE_GROUPS:
        for sex in ["Male", "Female"]:
            records.append({
                "age_group": ag,
                "sex": sex,
                "gq_population": 50.0,
            })
    return pd.DataFrame(records)


@pytest.fixture
def gq_county_all_ages():
    """Synthetic GQ data for one county, all 18 age groups x 2 sexes."""
    records = []
    for ag in AGE_GROUP_RANGES:
        for sex in ["Male", "Female"]:
            records.append({
                "age_group": ag,
                "sex": sex,
                "gq_population": 10.0,
            })
    return pd.DataFrame(records)


@pytest.fixture
def base_pop_simple(minimal_config):
    """Synthetic base population: 91 ages x 2 sexes x 6 races, uniform 100 each."""
    rows = []
    for age in range(91):
        for sex in ["Male", "Female"]:
            for race in STANDARD_RACES:
                rows.append({
                    "year": 2025,
                    "age": age,
                    "sex": sex,
                    "race": race,
                    "population": 100.0,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def base_pop_small():
    """Small base population: ages 0-4 only, 2 sexes, 2 races, uniform 500."""
    rows = []
    for age in range(5):
        for sex in ["Male", "Female"]:
            for race in ["White alone, Non-Hispanic", "Hispanic (any race)"]:
                rows.append({
                    "year": 2025,
                    "age": age,
                    "sex": sex,
                    "race": race,
                    "population": 500.0,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test _expand_gq_to_single_year_ages
# ---------------------------------------------------------------------------


class TestExpandGqToSingleYearAges:
    """Tests for _expand_gq_to_single_year_ages."""

    def test_total_population_preserved(self, gq_county_3ages):
        """Total GQ population must be identical before and after expansion."""
        total_before = gq_county_3ages["gq_population"].sum()
        expanded = _expand_gq_to_single_year_ages(gq_county_3ages)
        total_after = expanded["gq_population"].sum()
        assert total_after == pytest.approx(total_before, abs=1e-10)

    def test_uniform_split_within_5yr_group(self):
        """Each 5-year group should split evenly into 5 single-year ages."""
        gq = pd.DataFrame([{
            "age_group": "20-24",
            "sex": "Male",
            "gq_population": 100.0,
        }])
        expanded = _expand_gq_to_single_year_ages(gq)
        assert len(expanded) == 5
        assert set(expanded["age"]) == {20, 21, 22, 23, 24}
        for val in expanded["gq_population"]:
            assert val == pytest.approx(20.0)

    def test_85_plus_group_expands_to_6_ages(self):
        """The 85+ terminal group should expand to ages 85-90 (6 single years)."""
        gq = pd.DataFrame([{
            "age_group": "85+",
            "sex": "Female",
            "gq_population": 60.0,
        }])
        expanded = _expand_gq_to_single_year_ages(gq)
        assert len(expanded) == 6
        assert set(expanded["age"]) == {85, 86, 87, 88, 89, 90}
        for val in expanded["gq_population"]:
            assert val == pytest.approx(10.0)

    def test_all_18_groups_produce_91_ages(self, gq_county_all_ages):
        """Expanding all 18 groups for one sex should produce ages 0-90 (91 values)."""
        male_only = gq_county_all_ages[gq_county_all_ages["sex"] == "Male"]
        expanded = _expand_gq_to_single_year_ages(male_only)
        assert set(expanded["age"]) == set(range(91))
        assert len(expanded) == 91

    def test_population_non_negative(self, gq_county_all_ages):
        """All expanded GQ populations must be >= 0."""
        expanded = _expand_gq_to_single_year_ages(gq_county_all_ages)
        assert (expanded["gq_population"] >= 0).all()

    def test_zero_gq_produces_zero_expansion(self):
        """A group with zero GQ should produce zero population for each single year."""
        gq = pd.DataFrame([{
            "age_group": "10-14",
            "sex": "Male",
            "gq_population": 0.0,
        }])
        expanded = _expand_gq_to_single_year_ages(gq)
        assert (expanded["gq_population"] == 0.0).all()
        assert len(expanded) == 5

    def test_output_columns(self, gq_county_3ages):
        """Output should have exactly [age, sex, gq_population] columns."""
        expanded = _expand_gq_to_single_year_ages(gq_county_3ages)
        assert set(expanded.columns) == {"age", "sex", "gq_population"}

    def test_unknown_age_group_skipped(self):
        """Unknown age groups are skipped (no rows produced, no crash)."""
        gq = pd.DataFrame([{
            "age_group": "INVALID",
            "sex": "Male",
            "gq_population": 100.0,
        }])
        expanded = _expand_gq_to_single_year_ages(gq)
        assert len(expanded) == 0

    def test_sex_preserved(self):
        """Sex labels should be preserved through expansion."""
        gq = pd.DataFrame([
            {"age_group": "0-4", "sex": "Male", "gq_population": 50.0},
            {"age_group": "0-4", "sex": "Female", "gq_population": 30.0},
        ])
        expanded = _expand_gq_to_single_year_ages(gq)
        male_rows = expanded[expanded["sex"] == "Male"]
        female_rows = expanded[expanded["sex"] == "Female"]
        assert len(male_rows) == 5
        assert len(female_rows) == 5
        assert male_rows["gq_population"].sum() == pytest.approx(50.0)
        assert female_rows["gq_population"].sum() == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Test _distribute_gq_across_races
# ---------------------------------------------------------------------------


class TestDistributeGqAcrossRaces:
    """Tests for _distribute_gq_across_races."""

    @pytest.fixture
    def gq_single_year_simple(self):
        """GQ single-year data: 2 ages, 2 sexes."""
        return pd.DataFrame([
            {"age": 20, "sex": "Male", "gq_population": 100.0},
            {"age": 20, "sex": "Female", "gq_population": 80.0},
            {"age": 21, "sex": "Male", "gq_population": 60.0},
            {"age": 21, "sex": "Female", "gq_population": 40.0},
        ])

    @pytest.fixture
    def base_pop_for_race_shares(self):
        """Base pop with known race proportions: White=60%, Hispanic=40%."""
        rows = []
        for age in range(91):
            for sex in ["Male", "Female"]:
                rows.append({
                    "year": 2025, "age": age, "sex": sex,
                    "race": "White alone, Non-Hispanic", "population": 60.0,
                })
                rows.append({
                    "year": 2025, "age": age, "sex": sex,
                    "race": "Hispanic (any race)", "population": 40.0,
                })
                # Zero for remaining races
                for race in STANDARD_RACES:
                    if race not in ("White alone, Non-Hispanic", "Hispanic (any race)"):
                        rows.append({
                            "year": 2025, "age": age, "sex": sex,
                            "race": race, "population": 0.0,
                        })
        return pd.DataFrame(rows)

    def test_total_gq_preserved_after_race_distribution(
        self, gq_single_year_simple, base_pop_for_race_shares, minimal_config
    ):
        """Sum of GQ across all races must equal original GQ for each age-sex cell."""
        result = _distribute_gq_across_races(
            gq_single_year_simple, base_pop_for_race_shares, minimal_config
        )
        # Group by age-sex and sum across races
        age_sex_totals = result.groupby(["age", "sex"])["gq_population"].sum()
        # Compare to original
        for _, row in gq_single_year_simple.iterrows():
            key = (row["age"], row["sex"])
            assert age_sex_totals[key] == pytest.approx(row["gq_population"], abs=1e-10)

    def test_race_shares_proportional_to_base_pop(
        self, gq_single_year_simple, base_pop_for_race_shares, minimal_config
    ):
        """GQ race distribution follows county race proportions (60/40 split)."""
        result = _distribute_gq_across_races(
            gq_single_year_simple, base_pop_for_race_shares, minimal_config
        )
        # Check Male age 20: total GQ = 100, expect White=60, Hispanic=40
        male_20 = result[(result["age"] == 20) & (result["sex"] == "Male")]
        white_gq = male_20[male_20["race"] == "White alone, Non-Hispanic"]["gq_population"].values[0]
        hispanic_gq = male_20[male_20["race"] == "Hispanic (any race)"]["gq_population"].values[0]
        assert white_gq == pytest.approx(60.0, abs=1e-10)
        assert hispanic_gq == pytest.approx(40.0, abs=1e-10)

    def test_zero_race_gets_zero_gq(
        self, gq_single_year_simple, base_pop_for_race_shares, minimal_config
    ):
        """Races with zero population in base should get zero GQ allocation."""
        result = _distribute_gq_across_races(
            gq_single_year_simple, base_pop_for_race_shares, minimal_config
        )
        black_gq = result[result["race"] == "Black alone, Non-Hispanic"]["gq_population"]
        assert (black_gq == 0.0).all()

    def test_all_standard_races_present(
        self, gq_single_year_simple, base_pop_for_race_shares, minimal_config
    ):
        """Output should contain rows for all 6 standard race categories."""
        result = _distribute_gq_across_races(
            gq_single_year_simple, base_pop_for_race_shares, minimal_config
        )
        result_races = set(result["race"].unique())
        assert result_races == set(STANDARD_RACES)

    def test_output_columns(
        self, gq_single_year_simple, base_pop_for_race_shares, minimal_config
    ):
        """Output should have columns [age, sex, race, gq_population]."""
        result = _distribute_gq_across_races(
            gq_single_year_simple, base_pop_for_race_shares, minimal_config
        )
        assert set(result.columns) == {"age", "sex", "race", "gq_population"}

    def test_gq_non_negative(
        self, gq_single_year_simple, base_pop_for_race_shares, minimal_config
    ):
        """All distributed GQ values should be non-negative."""
        result = _distribute_gq_across_races(
            gq_single_year_simple, base_pop_for_race_shares, minimal_config
        )
        assert (result["gq_population"] >= 0).all()

    def test_zero_population_base_uniform_distribution(self, minimal_config):
        """When base pop is all zeros, GQ should be distributed uniformly across races."""
        gq_sy = pd.DataFrame([
            {"age": 0, "sex": "Male", "gq_population": 120.0},
        ])
        # All-zero base population
        rows = []
        for age in range(91):
            for sex in ["Male", "Female"]:
                for race in STANDARD_RACES:
                    rows.append({
                        "year": 2025, "age": age, "sex": sex,
                        "race": race, "population": 0.0,
                    })
        zero_base = pd.DataFrame(rows)

        result = _distribute_gq_across_races(gq_sy, zero_base, minimal_config)
        male_0 = result[(result["age"] == 0) & (result["sex"] == "Male")]
        expected_per_race = 120.0 / len(STANDARD_RACES)
        for _, row in male_0.iterrows():
            assert row["gq_population"] == pytest.approx(expected_per_race, abs=1e-10)

    def test_row_count(
        self, gq_single_year_simple, base_pop_for_race_shares, minimal_config
    ):
        """Output should have n_input_rows * n_races rows."""
        result = _distribute_gq_across_races(
            gq_single_year_simple, base_pop_for_race_shares, minimal_config
        )
        expected_rows = len(gq_single_year_simple) * len(STANDARD_RACES)
        assert len(result) == expected_rows


# ---------------------------------------------------------------------------
# Test get_county_gq_population
# ---------------------------------------------------------------------------


class TestGetCountyGqPopulation:
    """Tests for get_county_gq_population."""

    def test_returns_none_when_no_data_stored(self):
        """Should return None when no GQ data has been stored for the county."""
        result = get_county_gq_population("38001")
        assert result is None

    def test_retrieves_stored_gq_data(self):
        """Should return the DataFrame previously stored for a county."""
        dummy_gq = pd.DataFrame({
            "year": [2025],
            "age": [20],
            "sex": ["Male"],
            "race": ["White alone, Non-Hispanic"],
            "gq_population": [50.0],
        })
        _county_gq_populations["38001"] = dummy_gq
        result = get_county_gq_population("38001")
        assert result is not None
        pd.testing.assert_frame_equal(result, dummy_gq)

    def test_fips_zero_padded(self):
        """FIPS codes should be zero-padded to 5 digits."""
        dummy_gq = pd.DataFrame({
            "year": [2025],
            "age": [0],
            "sex": ["Female"],
            "race": ["Hispanic (any race)"],
            "gq_population": [10.0],
        })
        _county_gq_populations["00001"] = dummy_gq
        # Call with unpadded fips
        result = get_county_gq_population("1")
        assert result is not None

    def test_different_counties_independent(self):
        """Different counties should return independent data."""
        gq_a = pd.DataFrame({
            "year": [2025], "age": [0], "sex": ["Male"],
            "race": ["White alone, Non-Hispanic"], "gq_population": [100.0],
        })
        gq_b = pd.DataFrame({
            "year": [2025], "age": [0], "sex": ["Male"],
            "race": ["White alone, Non-Hispanic"], "gq_population": [200.0],
        })
        _county_gq_populations["38001"] = gq_a
        _county_gq_populations["38017"] = gq_b
        assert get_county_gq_population("38001")["gq_population"].iloc[0] == 100.0
        assert get_county_gq_population("38017")["gq_population"].iloc[0] == 200.0


# ---------------------------------------------------------------------------
# Test _separate_gq_from_base_population
# ---------------------------------------------------------------------------


class TestSeparateGqFromBasePopulation:
    """Tests for _separate_gq_from_base_population."""

    @pytest.fixture
    def gq_data_for_cache(self):
        """GQ data suitable for loading into the module cache.

        Contains data for county 38017 with 3 age groups x 2 sexes.
        """
        records = []
        for ag in THREE_AGE_GROUPS:
            for sex in ["Male", "Female"]:
                records.append({
                    "county_fips": "38017",
                    "age_group": ag,
                    "sex": sex,
                    "gq_population": 30.0,
                })
        return pd.DataFrame(records)

    def _inject_gq_cache(self, gq_data):
        """Inject GQ data into the module-level cache."""
        import cohort_projections.data.load.base_population_loader as bpl
        bpl._gq_data_cache = gq_data

    def test_hh_plus_gq_equals_total(self, base_pop_simple, minimal_config, gq_data_for_cache):
        """Household pop + GQ pop must equal total pop for every cell."""
        self._inject_gq_cache(gq_data_for_cache)
        hh_pop, gq_pop = _separate_gq_from_base_population(
            "38017", base_pop_simple, minimal_config
        )
        # Merge on index (aligned DataFrames)
        total_check = hh_pop["population"] + gq_pop["gq_population"]
        pd.testing.assert_series_equal(
            total_check, base_pop_simple["population"],
            check_names=False,
        )

    def test_household_pop_non_negative(self, base_pop_simple, minimal_config, gq_data_for_cache):
        """Household population should never be negative (clamped to 0)."""
        self._inject_gq_cache(gq_data_for_cache)
        hh_pop, _ = _separate_gq_from_base_population(
            "38017", base_pop_simple, minimal_config
        )
        assert (hh_pop["population"] >= 0).all()

    def test_gq_pop_non_negative(self, base_pop_simple, minimal_config, gq_data_for_cache):
        """GQ population should never be negative."""
        self._inject_gq_cache(gq_data_for_cache)
        _, gq_pop = _separate_gq_from_base_population(
            "38017", base_pop_simple, minimal_config
        )
        assert (gq_pop["gq_population"] >= 0).all()

    def test_gq_capped_at_total_population(self, minimal_config):
        """GQ should be capped at total population when GQ > total."""
        # Build base pop with very small population
        rows = []
        for age in range(91):
            for sex in ["Male", "Female"]:
                for race in STANDARD_RACES:
                    rows.append({
                        "year": 2025,
                        "age": age,
                        "sex": sex,
                        "race": race,
                        "population": 1.0,  # Very small
                    })
        tiny_base = pd.DataFrame(rows)

        # GQ data with large values that exceed base pop
        gq_records = []
        for ag in AGE_GROUP_RANGES:
            for sex in ["Male", "Female"]:
                gq_records.append({
                    "county_fips": "38017",
                    "age_group": ag,
                    "sex": sex,
                    "gq_population": 500.0,  # Much larger than base
                })
        self._inject_gq_cache(pd.DataFrame(gq_records))

        hh_pop, gq_pop = _separate_gq_from_base_population(
            "38017", tiny_base, minimal_config
        )
        # Household pop should be zero (not negative) where GQ exceeds total
        assert (hh_pop["population"] >= 0).all()
        # Conservation: hh + gq = total (GQ capped)
        total_check = hh_pop["population"] + gq_pop["gq_population"]
        pd.testing.assert_series_equal(
            total_check, tiny_base["population"],
            check_names=False,
        )

    def test_missing_county_returns_full_population(self, base_pop_simple, minimal_config):
        """County not in GQ data should return full population unchanged."""
        # Cache with data for a different county only
        other_county_gq = pd.DataFrame([{
            "county_fips": "38099",
            "age_group": "0-4",
            "sex": "Male",
            "gq_population": 100.0,
        }])
        self._inject_gq_cache(other_county_gq)

        hh_pop, gq_pop = _separate_gq_from_base_population(
            "38017", base_pop_simple, minimal_config
        )
        # Population unchanged
        pd.testing.assert_series_equal(
            hh_pop["population"],
            base_pop_simple["population"],
            check_names=False,
        )
        # GQ is all zeros
        assert (gq_pop["gq_population"] == 0.0).all()

    def test_gq_disabled_returns_full_population(self, base_pop_simple):
        """When GQ is disabled in config, population should be returned unchanged."""
        config_disabled = {
            "demographics": {
                "sex": ["Male", "Female"],
                "race_ethnicity": {"categories": STANDARD_RACES},
                "age_groups": {"min_age": 0, "max_age": 90},
            },
            "base_population": {
                "group_quarters": {"enabled": False},
            },
        }
        hh_pop, gq_pop = _separate_gq_from_base_population(
            "38017", base_pop_simple, config_disabled
        )
        pd.testing.assert_series_equal(
            hh_pop["population"],
            base_pop_simple["population"],
            check_names=False,
        )
        assert (gq_pop["gq_population"] == 0.0).all()

    def test_stores_gq_in_module_dict(self, base_pop_simple, minimal_config, gq_data_for_cache):
        """Function should store GQ data in _county_gq_populations for pipeline retrieval."""
        self._inject_gq_cache(gq_data_for_cache)
        _separate_gq_from_base_population("38017", base_pop_simple, minimal_config)
        stored = get_county_gq_population("38017")
        assert stored is not None
        assert "gq_population" in stored.columns

    def test_output_shape_matches_input(self, base_pop_simple, minimal_config, gq_data_for_cache):
        """Both output DataFrames should have same number of rows as input."""
        self._inject_gq_cache(gq_data_for_cache)
        hh_pop, gq_pop = _separate_gq_from_base_population(
            "38017", base_pop_simple, minimal_config
        )
        assert len(hh_pop) == len(base_pop_simple)
        assert len(gq_pop) == len(base_pop_simple)

    def test_zero_gq_county_returns_full_population(self, base_pop_simple, minimal_config):
        """County with zero GQ values should return full population unchanged."""
        zero_gq = pd.DataFrame([{
            "county_fips": "38017",
            "age_group": "0-4",
            "sex": "Male",
            "gq_population": 0.0,
        }])
        self._inject_gq_cache(zero_gq)

        hh_pop, gq_pop = _separate_gq_from_base_population(
            "38017", base_pop_simple, minimal_config
        )
        pd.testing.assert_series_equal(
            hh_pop["population"],
            base_pop_simple["population"],
            check_names=False,
        )

    def test_gq_subtraction_correctness(self, minimal_config):
        """Verify exact subtraction amounts for a controlled scenario."""
        # 5 ages, 1 sex, 1 race for simplicity
        rows = []
        for age in range(91):
            for sex in ["Male", "Female"]:
                for race in STANDARD_RACES:
                    rows.append({
                        "year": 2025,
                        "age": age,
                        "sex": sex,
                        "race": race,
                        "population": 1000.0,
                    })
        base = pd.DataFrame(rows)

        # GQ for one age group (20-24) with 250 total per sex
        gq = pd.DataFrame([
            {"county_fips": "38017", "age_group": "20-24", "sex": "Male", "gq_population": 250.0},
            {"county_fips": "38017", "age_group": "20-24", "sex": "Female", "gq_population": 250.0},
        ])
        self._inject_gq_cache(gq)

        hh_pop, gq_pop = _separate_gq_from_base_population("38017", base, minimal_config)

        # Ages 20-24 should have reduced population
        for age in range(20, 25):
            for sex in ["Male", "Female"]:
                age_sex_rows = hh_pop[
                    (hh_pop["age"] == age) & (hh_pop["sex"] == sex)
                ]
                # GQ per single year: 250/5 = 50 per sex
                # Distributed across races by proportion (all equal at 1000 each)
                # Each race gets 50/6 of GQ subtracted
                total_hh = age_sex_rows["population"].sum()
                # Total base for this age-sex: 1000 * 6 races = 6000
                # Total GQ for this age-sex: 50
                # HH should be 6000 - 50 = 5950
                assert total_hh == pytest.approx(5950.0, abs=0.1)

        # Ages outside 20-24 should be unchanged
        for age in [0, 10, 50, 90]:
            for sex in ["Male", "Female"]:
                age_sex_rows = hh_pop[
                    (hh_pop["age"] == age) & (hh_pop["sex"] == sex)
                ]
                total_hh = age_sex_rows["population"].sum()
                assert total_hh == pytest.approx(6000.0, abs=0.1)


# ---------------------------------------------------------------------------
# Test subtract_gq_from_populations (residual_migration.py, Phase 2)
# ---------------------------------------------------------------------------


class TestSubtractGqFromPopulations:
    """Tests for subtract_gq_from_populations (Phase 2 core function)."""

    @pytest.fixture
    def populations_3_years(self):
        """Population snapshots for 3 time points, 2 counties, 3 age groups."""
        counties = ["38001", "38017"]
        age_groups = ["0-4", "20-24", "65-69"]
        sexes = ["Male", "Female"]

        def make_pop(year, base):
            records = []
            for fips in counties:
                for ag in age_groups:
                    for sex in sexes:
                        records.append({
                            "county_fips": fips,
                            "age_group": ag,
                            "sex": sex,
                            "population": base,
                        })
            return pd.DataFrame(records)

        return {
            2000: make_pop(2000, 1000.0),
            2010: make_pop(2010, 1100.0),
            2020: make_pop(2020, 1200.0),
        }

    @pytest.fixture
    def gq_historical_3_years(self):
        """Historical GQ data matching 3 years, 2 counties, 3 age groups."""
        counties = ["38001", "38017"]
        age_groups = ["0-4", "20-24", "65-69"]
        sexes = ["Male", "Female"]
        years = [2000, 2010, 2020]

        records = []
        for year in years:
            for fips in counties:
                for ag in age_groups:
                    for sex in sexes:
                        records.append({
                            "county_fips": fips,
                            "year": year,
                            "age_group": ag,
                            "sex": sex,
                            "gq_population": 50.0,
                        })
        return pd.DataFrame(records)

    def test_household_pop_equals_total_minus_gq(
        self, populations_3_years, gq_historical_3_years
    ):
        """For every year, HH pop = total pop - GQ pop."""
        result = subtract_gq_from_populations(populations_3_years, gq_historical_3_years)
        for year in populations_3_years:
            original_total = populations_3_years[year]["population"].sum()
            hh_total = result[year]["population"].sum()
            gq_total = gq_historical_3_years[
                gq_historical_3_years["year"] == year
            ]["gq_population"].sum()
            assert hh_total == pytest.approx(original_total - gq_total, abs=1e-6)

    def test_population_non_negative(
        self, populations_3_years, gq_historical_3_years
    ):
        """All household populations should be >= 0 (clipped)."""
        result = subtract_gq_from_populations(populations_3_years, gq_historical_3_years)
        for year in result:
            assert (result[year]["population"] >= 0).all()

    def test_output_keys_match_input(
        self, populations_3_years, gq_historical_3_years
    ):
        """Output dict should have same year keys as input."""
        result = subtract_gq_from_populations(populations_3_years, gq_historical_3_years)
        assert set(result.keys()) == set(populations_3_years.keys())

    def test_output_columns(
        self, populations_3_years, gq_historical_3_years
    ):
        """Output DataFrames should have [county_fips, age_group, sex, population]."""
        result = subtract_gq_from_populations(populations_3_years, gq_historical_3_years)
        expected_cols = {"county_fips", "age_group", "sex", "population"}
        for year in result:
            assert set(result[year].columns) == expected_cols

    def test_missing_gq_year_returns_unchanged(self, populations_3_years):
        """Year with no GQ data should return population unchanged."""
        # Empty GQ historical
        gq_empty = pd.DataFrame(columns=["county_fips", "year", "age_group", "sex", "gq_population"])
        result = subtract_gq_from_populations(populations_3_years, gq_empty)
        for year in populations_3_years:
            pd.testing.assert_series_equal(
                result[year]["population"].reset_index(drop=True),
                populations_3_years[year]["population"].reset_index(drop=True),
                check_names=False,
            )

    def test_gq_exceeding_population_clipped_to_zero(self):
        """When GQ > population, household should be clipped to 0."""
        pop = {
            2020: pd.DataFrame([{
                "county_fips": "38001",
                "age_group": "20-24",
                "sex": "Male",
                "population": 100.0,
            }]),
        }
        gq = pd.DataFrame([{
            "county_fips": "38001",
            "year": 2020,
            "age_group": "20-24",
            "sex": "Male",
            "gq_population": 500.0,  # Exceeds population
        }])
        result = subtract_gq_from_populations(pop, gq)
        assert result[2020]["population"].iloc[0] == 0.0

    def test_partial_gq_match_fills_na_with_zero(self):
        """Cells without GQ match should retain full population (fillna(0))."""
        pop = {
            2020: pd.DataFrame([
                {"county_fips": "38001", "age_group": "0-4", "sex": "Male", "population": 500.0},
                {"county_fips": "38001", "age_group": "20-24", "sex": "Male", "population": 300.0},
            ]),
        }
        gq = pd.DataFrame([{
            "county_fips": "38001",
            "year": 2020,
            "age_group": "20-24",
            "sex": "Male",
            "gq_population": 100.0,
        }])
        result = subtract_gq_from_populations(pop, gq)
        result_df = result[2020].sort_values("age_group").reset_index(drop=True)
        # 0-4 should be unchanged (no GQ match)
        assert result_df[result_df["age_group"] == "0-4"]["population"].iloc[0] == 500.0
        # 20-24 should be reduced by 100
        assert result_df[result_df["age_group"] == "20-24"]["population"].iloc[0] == 200.0

    def test_row_count_preserved(
        self, populations_3_years, gq_historical_3_years
    ):
        """Number of rows in each year's DataFrame should be unchanged."""
        result = subtract_gq_from_populations(populations_3_years, gq_historical_3_years)
        for year in populations_3_years:
            assert len(result[year]) == len(populations_3_years[year])

    def test_exact_subtraction_values(self):
        """Verify exact arithmetic for a simple known case."""
        pop = {
            2020: pd.DataFrame([
                {"county_fips": "38001", "age_group": "0-4", "sex": "Male", "population": 1000.0},
                {"county_fips": "38001", "age_group": "0-4", "sex": "Female", "population": 950.0},
            ]),
        }
        gq = pd.DataFrame([
            {"county_fips": "38001", "year": 2020, "age_group": "0-4", "sex": "Male", "gq_population": 123.4},
            {"county_fips": "38001", "year": 2020, "age_group": "0-4", "sex": "Female", "gq_population": 56.7},
        ])
        result = subtract_gq_from_populations(pop, gq)
        result_df = result[2020].sort_values("sex").reset_index(drop=True)
        female_row = result_df[result_df["sex"] == "Female"]
        male_row = result_df[result_df["sex"] == "Male"]
        assert female_row["population"].iloc[0] == pytest.approx(893.3, abs=1e-6)
        assert male_row["population"].iloc[0] == pytest.approx(876.6, abs=1e-6)

    def test_no_gq_column_in_output(
        self, populations_3_years, gq_historical_3_years
    ):
        """The gq_population column should be dropped from the output."""
        result = subtract_gq_from_populations(populations_3_years, gq_historical_3_years)
        for year in result:
            assert "gq_population" not in result[year].columns

    def test_multiple_counties_independent(self):
        """GQ subtraction for different counties should be independent."""
        pop = {
            2020: pd.DataFrame([
                {"county_fips": "38001", "age_group": "0-4", "sex": "Male", "population": 1000.0},
                {"county_fips": "38017", "age_group": "0-4", "sex": "Male", "population": 5000.0},
            ]),
        }
        gq = pd.DataFrame([
            {"county_fips": "38001", "year": 2020, "age_group": "0-4", "sex": "Male", "gq_population": 100.0},
            {"county_fips": "38017", "year": 2020, "age_group": "0-4", "sex": "Male", "gq_population": 200.0},
        ])
        result = subtract_gq_from_populations(pop, gq)
        r = result[2020]
        assert r[r["county_fips"] == "38001"]["population"].iloc[0] == pytest.approx(900.0)
        assert r[r["county_fips"] == "38017"]["population"].iloc[0] == pytest.approx(4800.0)
