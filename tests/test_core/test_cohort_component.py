"""
Unit tests for the cohort component projection module.

Tests the CohortComponentProjection class including initialization,
single-year projections, multi-year projections, and utility methods.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from cohort_projections.core.cohort_component import CohortComponentProjection


class TestCohortComponentProjectionFixtures:
    """Shared fixtures for cohort component tests."""

    @pytest.fixture
    def sample_base_population(self):
        """Sample base population data for testing."""
        data = []
        # Create population for ages 0-90 by sex and race
        for sex in ["Male", "Female"]:
            for race in ["White", "Black"]:
                for age in range(91):
                    # Realistic age distribution (pyramid shape)
                    if age < 20:
                        pop = 1000.0
                    elif age < 65:
                        pop = 1200.0
                    else:
                        pop = 800.0 - (age - 65) * 20
                        pop = max(pop, 100.0)
                    data.append(
                        {
                            "year": 2025,
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "population": pop,
                        }
                    )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_fertility_rates(self):
        """Sample fertility rates for testing."""
        data = []
        for race in ["White", "Black"]:
            for age in range(15, 50):
                if 20 <= age <= 34:
                    rate = 0.08
                elif 15 <= age <= 19 or 35 <= age <= 39:
                    rate = 0.04
                else:
                    rate = 0.01
                data.append(
                    {
                        "age": age,
                        "race": race,
                        "fertility_rate": rate,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_survival_rates(self):
        """Sample survival rates for testing."""
        data = []
        for sex in ["Male", "Female"]:
            for race in ["White", "Black"]:
                for age in range(91):
                    if age < 1:
                        rate = 0.993
                    elif age < 65:
                        rate = 0.999
                    elif age < 80:
                        rate = 0.97
                    else:
                        rate = 0.90
                    data.append(
                        {
                            "age": age,
                            "sex": sex,
                            "race": race,
                            "survival_rate": rate,
                        }
                    )
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_migration_rates(self):
        """Sample migration rates for testing."""
        data = []
        for sex in ["Male", "Female"]:
            for race in ["White", "Black"]:
                for age in range(91):
                    if 20 <= age <= 35:
                        migration = 20.0
                    elif age >= 65:
                        migration = -10.0
                    else:
                        migration = 5.0
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
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "project": {
                "base_year": 2025,
                "projection_horizon": 10,
            },
            "demographics": {
                "age_groups": {
                    "min_age": 0,
                    "max_age": 90,
                    "single_year": True,
                }
            },
            "rates": {
                "fertility": {
                    "apply_to_ages": [15, 49],
                    "sex_ratio_male": 0.51,
                },
                "mortality": {
                    "improvement_factor": 0.0,
                },
            },
            "scenarios": {
                "baseline": {
                    "fertility": "constant",
                    "mortality": "constant",
                    "migration": "recent_average",
                },
                "high_growth": {
                    "fertility": "+10_percent",
                    "mortality": "constant",
                    "migration": "+25_percent",
                },
            },
        }


class TestCohortComponentProjectionInitialization(TestCohortComponentProjectionFixtures):
    """Tests for CohortComponentProjection initialization."""

    def test_initialization_valid_inputs(
        self,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Test initialization with valid inputs."""
        projection = CohortComponentProjection(
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
        )

        assert projection is not None
        assert projection.base_year == 2025
        assert projection.projection_horizon == 10
        assert projection.max_age == 90

    def test_initialization_stores_copies(
        self,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Test that initialization stores copies of input data."""
        projection = CohortComponentProjection(
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
        )

        # Modify original data
        sample_base_population.loc[0, "population"] = 99999.0

        # Projection's copy should be unchanged
        assert projection.base_population.loc[0, "population"] != 99999.0

    def test_initialization_empty_population_raises_error(
        self,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Test that empty base population raises ValueError."""
        empty_pop = pd.DataFrame(columns=["year", "age", "sex", "race", "population"])

        with pytest.raises(ValueError, match="empty"):
            CohortComponentProjection(
                base_population=empty_pop,
                fertility_rates=sample_fertility_rates,
                survival_rates=sample_survival_rates,
                migration_rates=sample_migration_rates,
                config=sample_config,
            )

    def test_initialization_missing_columns_raises_error(
        self,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Test that missing columns raise ValueError."""
        invalid_pop = pd.DataFrame(
            {
                "year": [2025],
                "age": [25],
                "sex": ["Male"],
                # Missing 'race' and 'population'
            }
        )

        with pytest.raises(ValueError, match="missing columns"):
            CohortComponentProjection(
                base_population=invalid_pop,
                fertility_rates=sample_fertility_rates,
                survival_rates=sample_survival_rates,
                migration_rates=sample_migration_rates,
                config=sample_config,
            )

    def test_initialization_negative_population_raises_error(
        self,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Test that negative population values raise ValueError."""
        sample_base_population.loc[0, "population"] = -100.0

        with pytest.raises(ValueError, match="negative"):
            CohortComponentProjection(
                base_population=sample_base_population,
                fertility_rates=sample_fertility_rates,
                survival_rates=sample_survival_rates,
                migration_rates=sample_migration_rates,
                config=sample_config,
            )


class TestProjectSingleYear(TestCohortComponentProjectionFixtures):
    """Tests for project_single_year method."""

    @pytest.fixture
    def projection(
        self,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Create a projection instance for testing."""
        return CohortComponentProjection(
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
        )

    def test_project_single_year_returns_dataframe(self, projection):
        """Test that project_single_year returns a DataFrame."""
        result = projection.project_single_year(projection.base_population, year=2025)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_project_single_year_correct_columns(self, projection):
        """Test that result has correct columns."""
        result = projection.project_single_year(projection.base_population, year=2025)

        expected_columns = ["year", "age", "sex", "race", "population"]
        assert all(col in result.columns for col in expected_columns)

    def test_project_single_year_increments_year(self, projection):
        """Test that year is incremented by 1."""
        result = projection.project_single_year(projection.base_population, year=2025)

        assert (result["year"] == 2026).all()

    def test_project_single_year_includes_births(self, projection):
        """Test that projection includes newborns (age 0)."""
        result = projection.project_single_year(projection.base_population, year=2025)

        age_0_pop = result[result["age"] == 0]["population"].sum()
        assert age_0_pop > 0

    def test_project_single_year_preserves_categories(self, projection):
        """Test that sex and race categories are preserved."""
        result = projection.project_single_year(projection.base_population, year=2025)

        original_sexes = set(projection.base_population["sex"].unique())
        original_races = set(projection.base_population["race"].unique())

        assert set(result["sex"].unique()) == original_sexes
        assert set(result["race"].unique()) == original_races

    def test_project_single_year_with_scenario(self, projection):
        """Test projection with scenario adjustment."""
        result_baseline = projection.project_single_year(
            projection.base_population, year=2025, scenario="baseline"
        )

        result_high = projection.project_single_year(
            projection.base_population, year=2025, scenario="high_growth"
        )

        # High growth should have more population due to higher fertility and migration
        total_baseline = result_baseline["population"].sum()
        total_high = result_high["population"].sum()

        assert total_high > total_baseline

    def test_project_single_year_population_change(self, projection):
        """Test that population changes as expected."""
        result = projection.project_single_year(projection.base_population, year=2025)

        total_before = projection.base_population["population"].sum()
        total_after = result["population"].sum()

        # Population should change due to births, deaths, and migration
        assert total_before != total_after


class TestRunProjection(TestCohortComponentProjectionFixtures):
    """Tests for run_projection method."""

    @pytest.fixture
    def projection(
        self,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Create a projection instance for testing."""
        return CohortComponentProjection(
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
        )

    def test_run_projection_returns_dataframe(self, projection):
        """Test that run_projection returns a DataFrame."""
        result = projection.run_projection(start_year=2025, end_year=2027)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_run_projection_includes_all_years(self, projection):
        """Test that result includes all projection years."""
        result = projection.run_projection(start_year=2025, end_year=2028)

        years_in_result = sorted(result["year"].unique())
        expected_years = [2025, 2026, 2027, 2028]

        assert years_in_result == expected_years

    def test_run_projection_stores_results(self, projection):
        """Test that results are stored in projection_results attribute."""
        result = projection.run_projection(start_year=2025, end_year=2027)

        assert not projection.projection_results.empty
        pd.testing.assert_frame_equal(result, projection.projection_results)

    def test_run_projection_creates_summaries(self, projection):
        """Test that annual summaries are created."""
        projection.run_projection(start_year=2025, end_year=2028)

        assert len(projection.annual_summaries) > 0

    def test_run_projection_default_parameters(self, projection):
        """Test run_projection with default parameters."""
        result = projection.run_projection()

        # Should use base_year and projection_horizon from config
        years_in_result = result["year"].unique()
        assert 2025 in years_in_result
        assert 2035 in years_in_result  # 2025 + 10

    def test_run_projection_population_trend(self, projection):
        """Test that population changes over projection period."""
        result = projection.run_projection(start_year=2025, end_year=2030)

        pop_2025 = result[result["year"] == 2025]["population"].sum()
        pop_2030 = result[result["year"] == 2030]["population"].sum()

        # Population should have changed
        assert pop_2025 != pop_2030


class TestUtilityMethods(TestCohortComponentProjectionFixtures):
    """Tests for utility methods."""

    @pytest.fixture
    def projection_with_results(
        self,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Create a projection instance with results."""
        projection = CohortComponentProjection(
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
        )
        projection.run_projection(start_year=2025, end_year=2030)
        return projection

    def test_get_projection_summary(self, projection_with_results):
        """Test get_projection_summary method."""
        summary = projection_with_results.get_projection_summary()

        assert isinstance(summary, pd.DataFrame)
        assert not summary.empty
        assert "year" in summary.columns
        assert "total_population" in summary.columns

    def test_get_population_by_year(self, projection_with_results):
        """Test get_population_by_year method."""
        year_data = projection_with_results.get_population_by_year(2027)

        assert isinstance(year_data, pd.DataFrame)
        assert not year_data.empty
        assert (year_data["year"] == 2027).all()

    def test_get_population_by_year_missing(self, projection_with_results):
        """Test get_population_by_year with non-existent year."""
        year_data = projection_with_results.get_population_by_year(2050)

        assert year_data.empty

    def test_get_cohort_trajectory(self, projection_with_results):
        """Test get_cohort_trajectory method."""
        # Track a cohort born in 2025
        trajectory = projection_with_results.get_cohort_trajectory(
            birth_year=2000, sex="Male", race="White"
        )

        assert isinstance(trajectory, pd.DataFrame)
        if not trajectory.empty:
            # Ages should increase by 1 each year
            ages = trajectory.sort_values("year")["age"].tolist()
            for i in range(1, len(ages)):
                assert ages[i] == ages[i - 1] + 1


class TestExportMethods(TestCohortComponentProjectionFixtures):
    """Tests for export methods."""

    @pytest.fixture
    def projection_with_results(
        self,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Create a projection instance with results."""
        projection = CohortComponentProjection(
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
        )
        projection.run_projection(start_year=2025, end_year=2027)
        return projection

    def test_export_results_parquet(self, projection_with_results):
        """Test exporting results to parquet format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.parquet"
            projection_with_results.export_results(output_path, format="parquet")

            assert output_path.exists()

            # Verify content
            loaded = pd.read_parquet(output_path)
            assert len(loaded) == len(projection_with_results.projection_results)

    def test_export_results_csv(self, projection_with_results):
        """Test exporting results to CSV format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            projection_with_results.export_results(output_path, format="csv")

            assert output_path.exists()

            # Verify content
            loaded = pd.read_csv(output_path)
            assert len(loaded) == len(projection_with_results.projection_results)

    def test_export_summary_csv(self, projection_with_results):
        """Test exporting summary to CSV format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "summary.csv"
            projection_with_results.export_summary(output_path, format="csv")

            assert output_path.exists()

    def test_export_results_invalid_format_raises_error(self, projection_with_results):
        """Test that invalid export format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.xyz"

            with pytest.raises(ValueError, match="Unsupported format"):
                projection_with_results.export_results(output_path, format="xyz")


class TestAnnualSummaryCalculations(TestCohortComponentProjectionFixtures):
    """Tests for annual summary calculations."""

    @pytest.fixture
    def projection_with_results(
        self,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Create a projection instance with results."""
        projection = CohortComponentProjection(
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
        )
        projection.run_projection(start_year=2025, end_year=2030)
        return projection

    def test_summary_has_total_population(self, projection_with_results):
        """Test that summary includes total population."""
        summary = projection_with_results.get_projection_summary()

        assert "total_population" in summary.columns
        assert (summary["total_population"] > 0).all()

    def test_summary_has_population_by_sex(self, projection_with_results):
        """Test that summary includes population by sex."""
        summary = projection_with_results.get_projection_summary()

        assert "male_population" in summary.columns
        assert "female_population" in summary.columns

        # Male + female should approximately equal total
        for _, row in summary.iterrows():
            sex_total = row["male_population"] + row["female_population"]
            assert abs(sex_total - row["total_population"]) < 1.0

    def test_summary_has_age_structure(self, projection_with_results):
        """Test that summary includes age structure metrics."""
        summary = projection_with_results.get_projection_summary()

        assert "median_age" in summary.columns
        assert "dependency_ratio" in summary.columns
        assert "population_under_18" in summary.columns
        assert "population_working_age" in summary.columns
        assert "population_65_plus" in summary.columns

    def test_summary_median_age_reasonable(self, projection_with_results):
        """Test that median age is reasonable."""
        summary = projection_with_results.get_projection_summary()

        # Median age should be between 0 and 90
        assert (summary["median_age"] >= 0).all()
        assert (summary["median_age"] <= 90).all()

    def test_summary_dependency_ratio_reasonable(self, projection_with_results):
        """Test that dependency ratio is reasonable."""
        summary = projection_with_results.get_projection_summary()

        # Dependency ratio typically between 0.3 and 1.5
        assert (summary["dependency_ratio"] >= 0).all()
        assert (summary["dependency_ratio"] <= 2.0).all()


class TestEdgeCases(TestCohortComponentProjectionFixtures):
    """Test edge cases for cohort component projection."""

    def test_single_year_projection(
        self,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Test projection for a single year."""
        projection = CohortComponentProjection(
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
        )

        result = projection.run_projection(start_year=2025, end_year=2026)

        years = result["year"].unique()
        assert len(years) == 2  # 2025 and 2026

    def test_minimal_population(
        self,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Test with minimal population data."""
        minimal_pop = pd.DataFrame(
            {
                "year": [2025, 2025],
                "age": [25, 30],
                "sex": ["Male", "Female"],
                "race": ["White", "White"],
                "population": [1000.0, 1000.0],
            }
        )

        projection = CohortComponentProjection(
            base_population=minimal_pop,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
        )

        result = projection.run_projection(start_year=2025, end_year=2027)

        assert not result.empty

    def test_long_projection_horizon(
        self,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Test projection over a longer time horizon."""
        projection = CohortComponentProjection(
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
        )

        # Project for 20 years
        result = projection.run_projection(start_year=2025, end_year=2045)

        assert 2045 in result["year"].unique()
        assert len(projection.annual_summaries) == 20  # 20 years projected


class TestCalculateMedianAge:
    """Tests for _calculate_median_age helper method."""

    def test_median_age_uniform_distribution(self):
        """Test median age calculation with uniform age distribution."""
        pop = pd.DataFrame(
            {
                "age": [0, 10, 20, 30, 40],
                "sex": ["Male"] * 5,
                "race": ["White"] * 5,
                "population": [100.0, 100.0, 100.0, 100.0, 100.0],
            }
        )

        # Create minimal projection just to access the method
        base_pop = pd.DataFrame(
            {
                "year": [2025],
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "population": [1000.0],
            }
        )
        fertility = pd.DataFrame(
            {
                "age": [25],
                "race": ["White"],
                "fertility_rate": [0.08],
            }
        )
        survival = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "survival_rate": [0.999],
            }
        )
        migration = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "net_migration": [10.0],
            }
        )

        projection = CohortComponentProjection(
            base_population=base_pop,
            fertility_rates=fertility,
            survival_rates=survival,
            migration_rates=migration,
        )

        median = projection._calculate_median_age(pop)

        # With uniform distribution, median should be 20
        assert median == 20.0

    def test_median_age_empty_population(self):
        """Test median age with empty population returns 0."""
        empty_pop = pd.DataFrame(columns=["age", "sex", "race", "population"])

        base_pop = pd.DataFrame(
            {
                "year": [2025],
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "population": [1000.0],
            }
        )
        fertility = pd.DataFrame(
            {
                "age": [25],
                "race": ["White"],
                "fertility_rate": [0.08],
            }
        )
        survival = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "survival_rate": [0.999],
            }
        )
        migration = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "net_migration": [10.0],
            }
        )

        projection = CohortComponentProjection(
            base_population=base_pop,
            fertility_rates=fertility,
            survival_rates=survival,
            migration_rates=migration,
        )

        median = projection._calculate_median_age(empty_pop)

        assert median == 0.0


class TestCalculateDependencyRatio:
    """Tests for _calculate_dependency_ratio helper method."""

    def test_dependency_ratio_calculation(self):
        """Test dependency ratio calculation."""
        pop = pd.DataFrame(
            {
                "age": [10, 40, 70],  # Child, working-age, senior
                "sex": ["Male"] * 3,
                "race": ["White"] * 3,
                "population": [200.0, 500.0, 300.0],
            }
        )

        base_pop = pd.DataFrame(
            {
                "year": [2025],
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "population": [1000.0],
            }
        )
        fertility = pd.DataFrame(
            {
                "age": [25],
                "race": ["White"],
                "fertility_rate": [0.08],
            }
        )
        survival = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "survival_rate": [0.999],
            }
        )
        migration = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "net_migration": [10.0],
            }
        )

        projection = CohortComponentProjection(
            base_population=base_pop,
            fertility_rates=fertility,
            survival_rates=survival,
            migration_rates=migration,
        )

        ratio = projection._calculate_dependency_ratio(pop)

        # Dependents (200 + 300) / Working age (500) = 1.0
        assert ratio == 1.0

    def test_dependency_ratio_no_working_age(self):
        """Test dependency ratio when no working age population."""
        pop = pd.DataFrame(
            {
                "age": [10, 70],  # Only dependents
                "sex": ["Male"] * 2,
                "race": ["White"] * 2,
                "population": [500.0, 500.0],
            }
        )

        base_pop = pd.DataFrame(
            {
                "year": [2025],
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "population": [1000.0],
            }
        )
        fertility = pd.DataFrame(
            {
                "age": [25],
                "race": ["White"],
                "fertility_rate": [0.08],
            }
        )
        survival = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "survival_rate": [0.999],
            }
        )
        migration = pd.DataFrame(
            {
                "age": [25],
                "sex": ["Male"],
                "race": ["White"],
                "net_migration": [10.0],
            }
        )

        projection = CohortComponentProjection(
            base_population=base_pop,
            fertility_rates=fertility,
            survival_rates=survival,
            migration_rates=migration,
        )

        ratio = projection._calculate_dependency_ratio(pop)

        # Should return 0 when dividing by 0
        assert ratio == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
