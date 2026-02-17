"""
Tests for Phase 4 time-varying rate support in the projection engine.

Verifies that:
- The engine remains backward compatible when *_by_year params are None
- Time-varying migration rates are selected by year_offset
- Time-varying survival rates are selected by calendar year
- Missing years fall back to constant rates
- Both time-varying rate types work simultaneously
- The format bridge correctly expands 5-year age groups to engine format
"""

import pandas as pd
import pytest

from cohort_projections.core.cohort_component import CohortComponentProjection

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RACES = ["White", "Black"]
SEXES = ["Male", "Female"]


@pytest.fixture
def sample_config():
    """Minimal projection config for testing."""
    return {
        "project": {
            "base_year": 2025,
            "projection_horizon": 5,
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
        },
    }


@pytest.fixture
def sample_base_population():
    """Small base population for fast tests."""
    data = []
    for sex in SEXES:
        for race in RACES:
            for age in range(91):
                pop = 1000.0 if age < 65 else max(100.0, 800.0 - (age - 65) * 20)
                data.append({"year": 2025, "age": age, "sex": sex, "race": race, "population": pop})
    return pd.DataFrame(data)


@pytest.fixture
def sample_fertility_rates():
    """Fertility rates covering ages 15-49 for two races."""
    data = []
    for race in RACES:
        for age in range(15, 50):
            rate = 0.08 if 20 <= age <= 34 else 0.02
            data.append({"age": age, "race": race, "fertility_rate": rate})
    return pd.DataFrame(data)


@pytest.fixture
def sample_survival_rates():
    """Constant survival rates (the 'base' rates)."""
    data = []
    for sex in SEXES:
        for race in RACES:
            for age in range(91):
                rate = 0.999 if age < 65 else 0.97
                data.append({"age": age, "sex": sex, "race": race, "survival_rate": rate})
    return pd.DataFrame(data)


@pytest.fixture
def sample_migration_rates():
    """Constant migration rates (the 'base' rates)."""
    data = [
        {"age": age, "sex": sex, "race": race, "net_migration": 5.0}
        for sex in SEXES
        for race in RACES
        for age in range(91)
    ]
    return pd.DataFrame(data)


def _make_migration_by_year(base_migration: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Create time-varying migration dicts with doubled rates for year_offset 1."""
    doubled = base_migration.copy()
    doubled["net_migration"] = doubled["net_migration"] * 2.0

    tripled = base_migration.copy()
    tripled["net_migration"] = tripled["net_migration"] * 3.0

    return {1: doubled, 2: tripled}


def _make_survival_by_year(base_survival: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Create time-varying survival dicts with slightly lower rates for 2025."""
    lower = base_survival.copy()
    lower["survival_rate"] = lower["survival_rate"] * 0.99

    higher = base_survival.copy()
    higher["survival_rate"] = (higher["survival_rate"] * 1.001).clip(upper=1.0)

    return {2025: lower, 2026: higher}


# ---------------------------------------------------------------------------
# TestTimeVaryingEngine
# ---------------------------------------------------------------------------


class TestTimeVaryingEngine:
    """Tests for time-varying rate lookups in the projection engine."""

    def test_constant_rates_backward_compatible(
        self,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """No *_by_year params -> engine works exactly as before."""
        engine = CohortComponentProjection(
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
        )

        # Attributes default to None
        assert engine.migration_rates_by_year is None
        assert engine.survival_rates_by_year is None

        # Lookups return the constant base rates
        mig = engine._get_migration_rates(2025)
        surv = engine._get_survival_rates(2025)
        pd.testing.assert_frame_equal(mig, engine.migration_rates)
        pd.testing.assert_frame_equal(surv, engine.survival_rates)

        # Full projection still runs
        result = engine.run_projection(start_year=2025, end_year=2027)
        assert not result.empty
        assert set(result["year"].unique()) == {2025, 2026, 2027}

    def test_migration_varies_by_year(
        self,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Different migration rates returned for different year offsets."""
        mig_by_year = _make_migration_by_year(sample_migration_rates)

        engine = CohortComponentProjection(
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
            migration_rates_by_year=mig_by_year,
        )

        # year 2025 -> offset 1 -> doubled rates
        rates_y1 = engine._get_migration_rates(2025)
        assert (rates_y1["net_migration"] == sample_migration_rates["net_migration"] * 2.0).all()

        # year 2026 -> offset 2 -> tripled rates
        rates_y2 = engine._get_migration_rates(2026)
        assert (rates_y2["net_migration"] == sample_migration_rates["net_migration"] * 3.0).all()

    def test_survival_varies_by_year(
        self,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Different survival rates returned for different calendar years."""
        surv_by_year = _make_survival_by_year(sample_survival_rates)

        engine = CohortComponentProjection(
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
            survival_rates_by_year=surv_by_year,
        )

        rates_2025 = engine._get_survival_rates(2025)
        rates_2026 = engine._get_survival_rates(2026)

        # 2025 should have lower survival than 2026
        assert rates_2025["survival_rate"].mean() < rates_2026["survival_rate"].mean()

        # Both should differ from the constant base
        assert not rates_2025["survival_rate"].equals(sample_survival_rates["survival_rate"])

    def test_missing_year_falls_back_to_constant(
        self,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Year not in dict -> uses base rates."""
        # Only provide offset 1 for migration
        mig_by_year = {1: sample_migration_rates.copy()}
        # Only provide year 2025 for survival
        surv_by_year = {2025: sample_survival_rates.copy()}

        engine = CohortComponentProjection(
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
            migration_rates_by_year=mig_by_year,
            survival_rates_by_year=surv_by_year,
        )

        # Offset 1 exists -> returns provided rates
        assert engine._get_migration_rates(2025) is mig_by_year[1]

        # Offset 5 does NOT exist -> falls back to constant
        fallback_mig = engine._get_migration_rates(2029)  # offset = 5
        pd.testing.assert_frame_equal(fallback_mig, engine.migration_rates)

        # Year 2025 exists -> returns provided rates
        assert engine._get_survival_rates(2025) is surv_by_year[2025]

        # Year 2030 does NOT exist -> falls back to constant
        fallback_surv = engine._get_survival_rates(2030)
        pd.testing.assert_frame_equal(fallback_surv, engine.survival_rates)

    def test_combined_time_varying_migration_and_survival(
        self,
        sample_base_population,
        sample_fertility_rates,
        sample_survival_rates,
        sample_migration_rates,
        sample_config,
    ):
        """Both time-varying migration and survival provided simultaneously."""
        mig_by_year = _make_migration_by_year(sample_migration_rates)
        surv_by_year = _make_survival_by_year(sample_survival_rates)

        engine = CohortComponentProjection(
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
            migration_rates_by_year=mig_by_year,
            survival_rates_by_year=surv_by_year,
        )

        # Projection should succeed with both time-varying rate types
        result = engine.run_projection(start_year=2025, end_year=2027)
        assert not result.empty
        assert set(result["year"].unique()) == {2025, 2026, 2027}

        # Compare to a constant-rate run to verify rates had an effect
        engine_constant = CohortComponentProjection(
            base_population=sample_base_population,
            fertility_rates=sample_fertility_rates,
            survival_rates=sample_survival_rates,
            migration_rates=sample_migration_rates,
            config=sample_config,
        )
        result_constant = engine_constant.run_projection(start_year=2025, end_year=2027)

        # Populations should differ because rates differ
        pop_tv = result[result["year"] == 2027]["population"].sum()
        pop_const = result_constant[result_constant["year"] == 2027]["population"].sum()
        assert pop_tv != pop_const

    def test_time_varying_survival_is_not_double_improved(self):
        """Per-year survival tables should not get config improvement applied again."""
        config = {
            "project": {
                "base_year": 2025,
                "projection_horizon": 20,
            },
            "demographics": {
                "age_groups": {
                    "max_age": 90,
                }
            },
            "rates": {
                "fertility": {
                    "apply_to_ages": [15, 49],
                    "sex_ratio_male": 0.51,
                },
                "mortality": {
                    "improvement_factor": 0.05,
                },
            },
            "scenarios": {
                "baseline": {
                    "fertility": "constant",
                    "mortality": "constant",
                    "migration": "recent_average",
                },
            },
        }

        base_population = pd.DataFrame(
            [{"year": 2025, "age": 10, "sex": "Male", "race": "White", "population": 100.0}]
        )
        fertility_rates = pd.DataFrame(columns=["age", "race", "fertility_rate"])
        survival_rates = pd.DataFrame(
            [{"age": 10, "sex": "Male", "race": "White", "survival_rate": 0.90}]
        )
        migration_rates = pd.DataFrame(
            [{"age": 11, "sex": "Male", "race": "White", "migration_rate": 0.0}]
        )
        survival_by_year = {
            2026: pd.DataFrame([{"age": 10, "sex": "Male", "race": "White", "survival_rate": 0.90}])
        }

        engine = CohortComponentProjection(
            base_population=base_population,
            fertility_rates=fertility_rates,
            survival_rates=survival_rates,
            migration_rates=migration_rates,
            config=config,
            survival_rates_by_year=survival_by_year,
        )

        projected = engine.project_single_year(base_population, year=2026, scenario="baseline")
        survived = projected[(projected["age"] == 11) & (projected["sex"] == "Male")].iloc[0]

        # If improvement were incorrectly applied again, this would be 90.5.
        assert survived["population"] == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# TestFormatBridge
# ---------------------------------------------------------------------------

import importlib

# The pipeline script has a numeric prefix (02_) which prevents normal import
# syntax. Use importlib to load the module by its dotted path.
_pipeline_mod = importlib.import_module("scripts.pipeline.02_run_projections")
expand_5yr_migration_to_engine_format = _pipeline_mod.expand_5yr_migration_to_engine_format


# Standard 5-year age groups used by Phase 1/2
_AGE_GROUPS = [
    "0-4",
    "5-9",
    "10-14",
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80-84",
    "85+",
]

ENGINE_RACES = [
    "White alone, Non-Hispanic",
    "Black alone, Non-Hispanic",
    "AIAN alone, Non-Hispanic",
    "Asian/PI alone, Non-Hispanic",
    "Two or more races, Non-Hispanic",
    "Hispanic (any race)",
]


@pytest.fixture
def sample_5yr_migration():
    """36-row migration DataFrame (18 age groups x 2 sexes)."""
    rows = [
        {"age_group": ag, "sex": sex, "migration_rate": 0.05}
        for sex in ["Male", "Female"]
        for ag in _AGE_GROUPS
    ]
    return pd.DataFrame(rows)


class TestFormatBridge:
    """Tests for expand_5yr_migration_to_engine_format bridge function."""

    def test_5yr_to_single_year_expansion(self, sample_5yr_migration):
        """36 rows (18 groups x 2 sexes) -> 1,092 rows (91 ages x 2 sexes x 6 races)."""
        result = expand_5yr_migration_to_engine_format(sample_5yr_migration)

        assert len(result) == 91 * 2 * 6  # 1,092
        assert set(result.columns) == {"age", "sex", "race", "migration_rate"}

    def test_race_distribution_uniform(self, sample_5yr_migration):
        """Same migration_rate for all races within an age-sex cell."""
        result = expand_5yr_migration_to_engine_format(sample_5yr_migration)

        # For a specific age-sex cell, all 6 races should have the same rate
        cell = result[(result["age"] == 25) & (result["sex"] == "Male")]
        assert len(cell) == 6
        assert cell["migration_rate"].nunique() == 1

    def test_total_migration_preserved(self, sample_5yr_migration):
        """Mean rate is preserved after expansion (uniform rate -> same mean)."""
        result = expand_5yr_migration_to_engine_format(sample_5yr_migration)

        # Input has uniform 0.05 rate; output should also have 0.05 for every row
        assert (result["migration_rate"] == 0.05).all()

    def test_85plus_expansion(self, sample_5yr_migration):
        """85+ group expands to ages 85-90 (6 single-year ages)."""
        result = expand_5yr_migration_to_engine_format(sample_5yr_migration)

        ages_85_plus = result[result["age"] >= 85]["age"].unique()
        assert set(ages_85_plus) == {85, 86, 87, 88, 89, 90}

    def test_age_range_complete(self, sample_5yr_migration):
        """Output covers ages 0-90 continuously."""
        result = expand_5yr_migration_to_engine_format(sample_5yr_migration)

        all_ages = sorted(result["age"].unique())
        assert all_ages == list(range(91))

    def test_all_races_present(self, sample_5yr_migration):
        """All 6 race categories are present in the output."""
        result = expand_5yr_migration_to_engine_format(sample_5yr_migration)

        assert set(result["race"].unique()) == set(ENGINE_RACES)

    def test_varying_rates_preserved(self):
        """Different rates per age group are preserved after expansion."""
        rows = []
        for sex in ["Male", "Female"]:
            for i, ag in enumerate(_AGE_GROUPS):
                rows.append({"age_group": ag, "sex": sex, "migration_rate": 0.01 * (i + 1)})
        df = pd.DataFrame(rows)

        result = expand_5yr_migration_to_engine_format(df)

        # Age 0 should have rate 0.01 (first group "0-4")
        rate_age0 = result[(result["age"] == 0) & (result["sex"] == "Male")]["migration_rate"].iloc[
            0
        ]
        assert rate_age0 == pytest.approx(0.01)

        # Age 85 should have rate 0.18 (last group "85+", index 17 -> 0.01*18)
        rate_age85 = result[(result["age"] == 85) & (result["sex"] == "Male")][
            "migration_rate"
        ].iloc[0]
        assert rate_age85 == pytest.approx(0.18)
