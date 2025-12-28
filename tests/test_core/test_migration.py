"""
Unit tests for the migration module.

Tests apply_migration, apply_migration_scenario, validate_migration_data,
distribute_international_migration, and combine_domestic_international functions.
"""

import pandas as pd
import pytest

from cohort_projections.core.migration import (
    apply_migration,
    apply_migration_scenario,
    combine_domestic_international,
    distribute_international_migration,
    validate_migration_data,
)


class TestApplyMigration:
    """Tests for apply_migration function."""

    @pytest.fixture
    def sample_population(self):
        """Sample population data for testing."""
        data = []
        for sex in ["Male", "Female"]:
            for race in ["White", "Black"]:
                for age in range(91):
                    data.append(
                        {
                            "year": 2025,
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "population": 1000.0,
                        }
                    )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_migration_rates_absolute(self):
        """Sample migration rates with absolute numbers."""
        data = []
        for sex in ["Male", "Female"]:
            for race in ["White", "Black"]:
                for age in range(91):
                    # Young adults have more migration, net positive
                    if 20 <= age <= 35:
                        migration = 50.0  # Net in-migration
                    elif age >= 65:
                        migration = -20.0  # Net out-migration (retirees leaving)
                    else:
                        migration = 10.0  # Slight in-migration
                    data.append(
                        {
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "net_migration": migration,
                        }
                    )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_migration_rates_proportional(self):
        """Sample migration rates as proportions."""
        data = []
        for sex in ["Male", "Female"]:
            for race in ["White", "Black"]:
                for age in range(91):
                    if 20 <= age <= 35:
                        rate = 0.02  # 2% net in-migration
                    elif age >= 65:
                        rate = -0.01  # 1% net out-migration
                    else:
                        rate = 0.005
                    data.append(
                        {
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "migration_rate": rate,
                        }
                    )
        return pd.DataFrame(data)

    def test_apply_migration_absolute_valid_input(
        self, sample_population, sample_migration_rates_absolute
    ):
        """Test apply_migration with absolute migration numbers."""
        result = apply_migration(sample_population, sample_migration_rates_absolute, year=2025)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert all(col in result.columns for col in ["year", "age", "sex", "race", "population"])

    def test_apply_migration_rate_valid_input(
        self, sample_population, sample_migration_rates_proportional
    ):
        """Test apply_migration with migration rates."""
        result = apply_migration(sample_population, sample_migration_rates_proportional, year=2025)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_apply_migration_changes_population(
        self, sample_population, sample_migration_rates_absolute
    ):
        """Test that migration changes total population."""
        result = apply_migration(sample_population, sample_migration_rates_absolute, year=2025)

        total_before = sample_population["population"].sum()
        total_after = result["population"].sum()

        # With net in-migration, total should increase
        net_migration = sample_migration_rates_absolute["net_migration"].sum()
        expected_change = net_migration

        actual_change = total_after - total_before
        assert abs(actual_change - expected_change) < 1.0

    def test_apply_migration_year_increment(
        self, sample_population, sample_migration_rates_absolute
    ):
        """Test that year is incremented by 1."""
        result = apply_migration(sample_population, sample_migration_rates_absolute, year=2025)

        assert (result["year"] == 2026).all()

    def test_apply_migration_empty_population(self, sample_migration_rates_absolute):
        """Test apply_migration with empty population."""
        empty_pop = pd.DataFrame(columns=["year", "age", "sex", "race", "population"])

        result = apply_migration(empty_pop, sample_migration_rates_absolute, year=2025)

        assert result.empty

    def test_apply_migration_prevents_negative_population(self, sample_population):
        """Test that negative population is prevented."""
        # Create migration that would cause negative population
        severe_outmigration = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "net_migration": [-2000.0],  # More out-migration than population
            }
        )

        # Filter population to just the affected cohort
        filtered_pop = sample_population[
            (sample_population["age"] == 25)
            & (sample_population["sex"] == "Male")
            & (sample_population["race"] == "White")
        ].copy()

        result = apply_migration(filtered_pop, severe_outmigration, year=2025)

        # Population should be 0, not negative
        assert (result["population"] >= 0).all()

    def test_apply_migration_missing_columns_raises_error(self, sample_migration_rates_absolute):
        """Test that missing population columns raise ValueError."""
        invalid_pop = pd.DataFrame(
            {
                "year": [2025],
                "age": [25],
                "sex": ["Male"],
                # Missing 'race' and 'population'
            }
        )

        with pytest.raises(ValueError, match="population must have columns"):
            apply_migration(invalid_pop, sample_migration_rates_absolute, year=2025)

    def test_apply_migration_no_migration_column_raises_error(self, sample_population):
        """Test that missing migration column raises ValueError."""
        invalid_rates = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                # Missing 'net_migration' and 'migration_rate'
            }
        )

        with pytest.raises(ValueError, match="net_migration.*migration_rate"):
            apply_migration(sample_population, invalid_rates, year=2025)

    def test_apply_migration_handles_missing_rates(self, sample_population):
        """Test that missing migration rates default to 0."""
        # Only provide rates for some cohorts
        partial_rates = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "net_migration": [100.0],
            }
        )

        result = apply_migration(sample_population, partial_rates, year=2025)

        # Should still produce output (cohorts without rates get 0 migration)
        assert not result.empty

    def test_apply_migration_preserves_categories(
        self, sample_population, sample_migration_rates_absolute
    ):
        """Test that sex and race categories are preserved."""
        result = apply_migration(sample_population, sample_migration_rates_absolute, year=2025)

        assert set(result["sex"].unique()) == set(sample_population["sex"].unique())
        assert set(result["race"].unique()) == set(sample_population["race"].unique())


class TestApplyMigrationScenario:
    """Tests for apply_migration_scenario function."""

    @pytest.fixture
    def base_migration_rates(self):
        """Base migration rates for scenario testing."""
        return pd.DataFrame(
            {
                "age": [20, 30, 40, 70],
                "sex": ["Male"] * 4,
                "race": ["White"] * 4,
                "net_migration": [100.0, 200.0, 50.0, -50.0],
            }
        )

    @pytest.fixture
    def base_migration_rates_proportional(self):
        """Base migration rates (proportional) for scenario testing."""
        return pd.DataFrame(
            {
                "age": [20, 30, 40, 70],
                "sex": ["Male"] * 4,
                "race": ["White"] * 4,
                "migration_rate": [0.02, 0.03, 0.01, -0.01],
            }
        )

    def test_constant_scenario(self, base_migration_rates):
        """Test constant/recent_average scenario returns unchanged rates."""
        result_constant = apply_migration_scenario(
            base_migration_rates, "constant", year=2030, base_year=2025
        )
        result_recent = apply_migration_scenario(
            base_migration_rates, "recent_average", year=2030, base_year=2025
        )

        pd.testing.assert_frame_equal(result_constant, base_migration_rates)
        pd.testing.assert_frame_equal(result_recent, base_migration_rates)

    def test_plus_25_percent_scenario(self, base_migration_rates):
        """Test +25% scenario increases migration by 25%."""
        result = apply_migration_scenario(
            base_migration_rates, "+25_percent", year=2030, base_year=2025
        )

        expected = base_migration_rates["net_migration"] * 1.25
        pd.testing.assert_series_equal(result["net_migration"], expected, check_names=False)

    def test_minus_25_percent_scenario(self, base_migration_rates):
        """Test -25% scenario decreases migration by 25%."""
        result = apply_migration_scenario(
            base_migration_rates, "-25_percent", year=2030, base_year=2025
        )

        expected = base_migration_rates["net_migration"] * 0.75
        pd.testing.assert_series_equal(result["net_migration"], expected, check_names=False)

    def test_zero_scenario(self, base_migration_rates):
        """Test zero scenario sets all migration to 0."""
        result = apply_migration_scenario(base_migration_rates, "zero", year=2030, base_year=2025)

        assert (result["net_migration"] == 0.0).all()

    def test_double_scenario(self, base_migration_rates):
        """Test double scenario doubles migration."""
        result = apply_migration_scenario(base_migration_rates, "double", year=2030, base_year=2025)

        expected = base_migration_rates["net_migration"] * 2.0
        pd.testing.assert_series_equal(result["net_migration"], expected, check_names=False)

    def test_half_scenario(self, base_migration_rates):
        """Test half scenario halves migration."""
        result = apply_migration_scenario(base_migration_rates, "half", year=2030, base_year=2025)

        expected = base_migration_rates["net_migration"] * 0.5
        pd.testing.assert_series_equal(result["net_migration"], expected, check_names=False)

    def test_unknown_scenario_uses_base(self, base_migration_rates):
        """Test unknown scenario returns unchanged rates."""
        result = apply_migration_scenario(
            base_migration_rates, "unknown_scenario", year=2030, base_year=2025
        )

        pd.testing.assert_frame_equal(result, base_migration_rates)

    def test_scenario_works_with_migration_rate_column(self, base_migration_rates_proportional):
        """Test scenario adjustments work with migration_rate column."""
        result = apply_migration_scenario(
            base_migration_rates_proportional, "+25_percent", year=2030, base_year=2025
        )

        expected = base_migration_rates_proportional["migration_rate"] * 1.25
        pd.testing.assert_series_equal(result["migration_rate"], expected, check_names=False)

    def test_scenario_preserves_other_columns(self, base_migration_rates):
        """Test that scenario preserves non-migration columns."""
        result = apply_migration_scenario(
            base_migration_rates, "+25_percent", year=2030, base_year=2025
        )

        assert list(result["age"]) == list(base_migration_rates["age"])
        assert list(result["sex"]) == list(base_migration_rates["sex"])
        assert list(result["race"]) == list(base_migration_rates["race"])


class TestValidateMigrationData:
    """Tests for validate_migration_data function."""

    @pytest.fixture
    def valid_migration_rates(self):
        """Valid migration rates for testing."""
        data = []
        for sex in ["Male", "Female"]:
            for race in ["White"]:
                for age in range(91):
                    migration = 50.0 if 20 <= age <= 35 else 10.0
                    data.append(
                        {
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "net_migration": migration,
                        }
                    )
        return pd.DataFrame(data)

    def test_validate_valid_rates(self, valid_migration_rates):
        """Test validation passes for valid rates."""
        is_valid, issues = validate_migration_data(valid_migration_rates)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_missing_migration_column(self):
        """Test validation fails when migration column is missing."""
        invalid_rates = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                # Missing net_migration and migration_rate
            }
        )

        is_valid, issues = validate_migration_data(invalid_rates)

        assert is_valid is False
        assert any("net_migration" in issue and "migration_rate" in issue for issue in issues)

    def test_validate_missing_id_columns(self):
        """Test validation fails when ID columns are missing."""
        invalid_rates = pd.DataFrame(
            {
                "net_migration": [100.0],
                # Missing age, sex, race
            }
        )

        is_valid, issues = validate_migration_data(invalid_rates)

        assert is_valid is False
        assert any("Missing required columns" in issue for issue in issues)

    def test_validate_extreme_absolute_migration(self):
        """Test validation warns about very large migration values."""
        extreme_rates = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "net_migration": [50000.0],  # Very large
            }
        )

        is_valid, issues = validate_migration_data(extreme_rates)

        assert is_valid is False
        assert any("Very large net migration" in issue for issue in issues)

    def test_validate_extreme_migration_rate(self):
        """Test validation catches extreme migration rates."""
        extreme_rates = pd.DataFrame(
            {
                "age": [25, 30],
                "sex": ["Male", "Male"],
                "race": ["White", "White"],
                "migration_rate": [0.6, -1.5],  # Extreme rates
            }
        )

        is_valid, issues = validate_migration_data(extreme_rates)

        assert is_valid is False
        assert any(
            "Migration rates > 1.0" in issue or "Migration rates < -1.0" in issue
            for issue in issues
        )

    def test_validate_with_population_negative_result(self):
        """Test validation catches migration that would cause negative population."""
        migration = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "net_migration": [-2000.0],  # More than population
            }
        )

        population = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "population": [1000.0],
            }
        )

        is_valid, issues = validate_migration_data(migration, population=population)

        assert is_valid is False
        assert any("negative population" in issue for issue in issues)

    def test_validate_missing_combinations(self):
        """Test validation catches missing age-sex-race combinations."""
        incomplete_rates = pd.DataFrame(
            {
                "age": [25, 25, 30],  # Missing 30 for Female
                "sex": ["Male", "Female", "Male"],
                "race": ["White", "White", "White"],
                "net_migration": [100.0, 100.0, 100.0],
            }
        )

        is_valid, issues = validate_migration_data(incomplete_rates)

        assert is_valid is False
        assert any("Missing age-sex-race combinations" in issue for issue in issues)


class TestDistributeInternationalMigration:
    """Tests for distribute_international_migration function."""

    @pytest.fixture
    def sample_population(self):
        """Sample population for distribution testing."""
        data = []
        for sex in ["Male", "Female"]:
            for age in [20, 30, 40, 50]:
                data.append(
                    {
                        "age": age,
                        "sex": sex,
                        "race": "White",
                        "population": 1000.0,
                    }
                )
        return pd.DataFrame(data)

    def test_distribute_proportional_to_population(self, sample_population):
        """Test that migration is distributed proportionally to population."""
        total_migration = 800.0

        result = distribute_international_migration(total_migration, sample_population)

        assert isinstance(result, pd.DataFrame)
        assert "net_migration" in result.columns

        # Total should equal input
        assert abs(result["net_migration"].sum() - total_migration) < 0.01

        # Each cohort should get equal share (all populations are equal)
        expected_per_cohort = total_migration / len(sample_population)
        assert all(abs(m - expected_per_cohort) < 0.01 for m in result["net_migration"])

    def test_distribute_with_custom_age_distribution(self, sample_population):
        """Test distribution with custom age weights."""
        total_migration = 1000.0

        age_distribution = pd.DataFrame(
            {
                "age": [20, 30, 40, 50],
                "sex": ["Male", "Male", "Male", "Male"],
                "race": ["White", "White", "White", "White"],
                "weight": [0.4, 0.3, 0.2, 0.1],  # More weight on younger ages
            }
        )

        # Filter population to match
        male_pop = sample_population[sample_population["sex"] == "Male"].copy()

        result = distribute_international_migration(total_migration, male_pop, age_distribution)

        # Check total
        assert abs(result["net_migration"].sum() - total_migration) < 0.01

    def test_distribute_negative_migration(self, sample_population):
        """Test distribution of negative (out) migration."""
        total_migration = -500.0

        result = distribute_international_migration(total_migration, sample_population)

        assert result["net_migration"].sum() < 0
        assert abs(result["net_migration"].sum() - total_migration) < 0.01

    def test_distribute_zero_population(self):
        """Test distribution with zero total population."""
        zero_pop = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "population": [0.0],
            }
        )

        result = distribute_international_migration(1000.0, zero_pop)

        # Should return zeros when population is zero
        assert (result["net_migration"] == 0.0).all()


class TestCombineDomesticInternational:
    """Tests for combine_domestic_international function."""

    @pytest.fixture
    def sample_domestic_migration(self):
        """Sample domestic migration data."""
        return pd.DataFrame(
            {
                "age": [20, 30, 40],
                "sex": ["Male", "Male", "Male"],
                "race": ["White", "White", "White"],
                "net_migration": [100.0, 50.0, -30.0],
            }
        )

    @pytest.fixture
    def sample_international_migration(self):
        """Sample international migration data."""
        return pd.DataFrame(
            {
                "age": [20, 30, 40],
                "sex": ["Male", "Male", "Male"],
                "race": ["White", "White", "White"],
                "net_migration": [50.0, 25.0, 10.0],
            }
        )

    def test_combine_valid_inputs(self, sample_domestic_migration, sample_international_migration):
        """Test combining domestic and international migration."""
        result = combine_domestic_international(
            sample_domestic_migration, sample_international_migration
        )

        assert isinstance(result, pd.DataFrame)
        assert "net_migration" in result.columns

    def test_combine_sums_correctly(
        self, sample_domestic_migration, sample_international_migration
    ):
        """Test that combined migration is the sum of components."""
        result = combine_domestic_international(
            sample_domestic_migration, sample_international_migration
        )

        expected = [150.0, 75.0, -20.0]  # 100+50, 50+25, -30+10
        actual = result.sort_values("age")["net_migration"].tolist()

        for exp, act in zip(expected, actual, strict=False):
            assert abs(exp - act) < 0.01

    def test_combine_with_missing_cohorts(self):
        """Test combining when cohorts don't fully overlap."""
        domestic = pd.DataFrame(
            {
                "age": [20, 30],
                "sex": ["Male", "Male"],
                "race": ["White", "White"],
                "net_migration": [100.0, 50.0],
            }
        )

        international = pd.DataFrame(
            {
                "age": [30, 40],  # Different ages
                "sex": ["Male", "Male"],
                "race": ["White", "White"],
                "net_migration": [25.0, 10.0],
            }
        )

        result = combine_domestic_international(domestic, international)

        # Should have 3 cohorts (20, 30, 40)
        assert len(result) == 3

        # Age 30 should have sum, others should have single component
        age_30 = result[result["age"] == 30]["net_migration"].iloc[0]
        assert abs(age_30 - 75.0) < 0.01  # 50 + 25

    def test_combine_missing_net_migration_column(self):
        """Test that missing net_migration column raises error."""
        invalid_domestic = pd.DataFrame(
            {
                "age": [20],
                "sex": ["Male"],
                "race": ["White"],
                # Missing net_migration
            }
        )

        valid_international = pd.DataFrame(
            {
                "age": [20],
                "sex": ["Male"],
                "race": ["White"],
                "net_migration": [50.0],
            }
        )

        with pytest.raises(ValueError, match="domestic_migration must have"):
            combine_domestic_international(invalid_domestic, valid_international)

    def test_combine_preserves_id_columns(
        self, sample_domestic_migration, sample_international_migration
    ):
        """Test that ID columns are preserved in result."""
        result = combine_domestic_international(
            sample_domestic_migration, sample_international_migration
        )

        assert "age" in result.columns
        assert "sex" in result.columns
        assert "race" in result.columns


class TestEdgeCases:
    """Test edge cases for migration module."""

    def test_single_cohort_migration(self):
        """Test migration with a single cohort."""
        single_pop = pd.DataFrame(
            {
                "year": [2025],
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "population": [1000.0],
            }
        )
        single_migration = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "net_migration": [100.0],
            }
        )

        result = apply_migration(single_pop, single_migration, year=2025)

        assert len(result) == 1
        assert abs(result["population"].iloc[0] - 1100.0) < 0.1

    def test_zero_migration(self):
        """Test with zero migration."""
        pop = pd.DataFrame(
            {
                "year": [2025],
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "population": [1000.0],
            }
        )
        zero_migration = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "net_migration": [0.0],
            }
        )

        result = apply_migration(pop, zero_migration, year=2025)

        assert result["population"].iloc[0] == 1000.0

    def test_large_in_migration(self):
        """Test with large in-migration."""
        pop = pd.DataFrame(
            {
                "year": [2025],
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "population": [1000.0],
            }
        )
        large_migration = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "net_migration": [5000.0],  # Large in-migration
            }
        )

        result = apply_migration(pop, large_migration, year=2025)

        assert result["population"].iloc[0] == 6000.0


@pytest.mark.parametrize(
    "scenario,expected_multiplier",
    [
        ("constant", 1.0),
        ("recent_average", 1.0),
        ("+25_percent", 1.25),
        ("-25_percent", 0.75),
        ("double", 2.0),
        ("half", 0.5),
    ],
)
def test_scenario_multipliers(scenario, expected_multiplier):
    """Parametrized test for scenario multipliers."""
    rates = pd.DataFrame(
        {
            "age": [25],
            "sex": ["Male"],
            "race": ["White"],
            "net_migration": [100.0],
        }
    )

    result = apply_migration_scenario(rates, scenario, year=2025, base_year=2025)

    expected = 100.0 * expected_multiplier
    assert abs(result["net_migration"].iloc[0] - expected) < 0.0001


def test_zero_scenario_sets_zero():
    """Test that zero scenario sets migration to zero."""
    rates = pd.DataFrame(
        {
            "age": [25, 30],
            "sex": ["Male", "Male"],
            "race": ["White", "White"],
            "net_migration": [100.0, -50.0],
        }
    )

    result = apply_migration_scenario(rates, "zero", year=2025, base_year=2025)

    assert (result["net_migration"] == 0.0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
