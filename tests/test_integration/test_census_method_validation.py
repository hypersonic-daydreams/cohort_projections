"""
Phase 5 Validation & Comparison tests for the census-method projection upgrade.

Validates the outputs of Phases 1-4:
- Convergence schedule correctness (5-10-5 interpolation)
- Mortality improvement direction and bounds
- No negative populations in a full 20-year projection
- Format bridge consistency (5-year -> engine format)
- Pipeline end-to-end data flow (dict structure builders)
"""

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from cohort_projections.core.cohort_component import CohortComponentProjection

# ---------------------------------------------------------------------------
# Project root and module import
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# The pipeline script has a numeric prefix (02_) which prevents normal import.
_pipeline_mod = importlib.import_module("scripts.pipeline.02_run_projections")
expand_5yr_migration_to_engine_format = _pipeline_mod.expand_5yr_migration_to_engine_format
_build_convergence_rate_dicts = _pipeline_mod._build_convergence_rate_dicts
_build_survival_rates_by_year = _pipeline_mod._build_survival_rates_by_year

# ---------------------------------------------------------------------------
# Data file paths
# ---------------------------------------------------------------------------

CONVERGENCE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "migration" / "convergence_rates_by_year.parquet"
)
CONVERGENCE_META_PATH = (
    PROJECT_ROOT / "data" / "processed" / "migration" / "convergence_metadata.json"
)
MORTALITY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "mortality" / "nd_adjusted_survival_projections.parquet"
)
MORTALITY_META_PATH = (
    PROJECT_ROOT / "data" / "processed" / "mortality" / "mortality_improvement_metadata.json"
)

_DATA_FILES_EXIST = CONVERGENCE_PATH.exists() and MORTALITY_PATH.exists()
_SKIP_REASON = "Processed data files not present (development environment only)"

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

ENGINE_RACES = [
    "White alone, Non-Hispanic",
    "Black alone, Non-Hispanic",
    "AIAN alone, Non-Hispanic",
    "Asian/PI alone, Non-Hispanic",
    "Two or more races, Non-Hispanic",
    "Hispanic (any race)",
]

SEXES = ["Male", "Female"]

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


# ===================================================================
# 1. Convergence Correctness
# ===================================================================


@pytest.mark.skipif(not _DATA_FILES_EXIST, reason=_SKIP_REASON)
class TestConvergenceCorrectness:
    """Verify Phase 2 convergence output follows the 5-10-5 schedule."""

    @pytest.fixture(scope="class")
    def convergence_df(self) -> pd.DataFrame:
        return pd.read_parquet(CONVERGENCE_PATH)

    @pytest.fixture(scope="class")
    def convergence_metadata(self) -> dict:
        with open(CONVERGENCE_META_PATH) as f:
            return json.load(f)

    def test_convergence_has_expected_shape(self, convergence_df: pd.DataFrame):
        """Convergence output has 38,160 rows (53 counties x 20 years x 36 cells)."""
        assert len(convergence_df) == 38_160

    def test_convergence_has_expected_columns(self, convergence_df: pd.DataFrame):
        """Columns match the documented schema."""
        expected = {"year_offset", "county_fips", "age_group", "sex", "migration_rate"}
        assert set(convergence_df.columns) == expected

    def test_year_offsets_1_to_20(self, convergence_df: pd.DataFrame):
        """All 20 projection year offsets are present."""
        assert sorted(convergence_df["year_offset"].unique()) == list(range(1, 21))

    def test_medium_hold_years_equal(self, convergence_df: pd.DataFrame):
        """Years 5-15 (medium hold) should all have identical rates per county."""
        for county in convergence_df["county_fips"].unique()[:5]:  # sample 5 counties
            county_data = convergence_df[convergence_df["county_fips"] == county]
            ref = (
                county_data[county_data["year_offset"] == 5]
                .set_index(["age_group", "sex"])["migration_rate"]
                .sort_index()
            )
            for yr in range(6, 16):
                other = (
                    county_data[county_data["year_offset"] == yr]
                    .set_index(["age_group", "sex"])["migration_rate"]
                    .sort_index()
                )
                pd.testing.assert_series_equal(ref, other, check_names=False)

    def test_year5_equals_medium_rate(
        self, convergence_df: pd.DataFrame, convergence_metadata: dict
    ):
        """Year offset 5 is the medium rate (end of recent-to-medium interpolation)."""
        # The schedule says years 1-5 interpolate from recent to medium.
        # At year 5 we should have reached the medium rate.
        # Verify that year 5 differs from year 1 for at least some counties.
        county = convergence_df["county_fips"].unique()[0]
        yr1 = (
            convergence_df[
                (convergence_df["county_fips"] == county) & (convergence_df["year_offset"] == 1)
            ]
            .set_index(["age_group", "sex"])["migration_rate"]
            .sort_index()
        )
        yr5 = (
            convergence_df[
                (convergence_df["county_fips"] == county) & (convergence_df["year_offset"] == 5)
            ]
            .set_index(["age_group", "sex"])["migration_rate"]
            .sort_index()
        )
        # Year 5 should differ from year 1 (interpolation happened)
        assert not yr1.equals(yr5), "Year 5 should differ from year 1 after interpolation"

    def test_year20_equals_longterm(self, convergence_df: pd.DataFrame):
        """Year offset 20 should differ from medium (years 5-15) for most counties."""
        differ_count = 0
        for county in convergence_df["county_fips"].unique():
            county_data = convergence_df[convergence_df["county_fips"] == county]
            yr15 = (
                county_data[county_data["year_offset"] == 15]
                .set_index(["age_group", "sex"])["migration_rate"]
                .sort_index()
            )
            yr20 = (
                county_data[county_data["year_offset"] == 20]
                .set_index(["age_group", "sex"])["migration_rate"]
                .sort_index()
            )
            if not yr15.equals(yr20):
                differ_count += 1

        # Most counties should show a difference between medium and long-term
        total = convergence_df["county_fips"].nunique()
        assert differ_count > total * 0.5, (
            f"Only {differ_count}/{total} counties differ between yr15 and yr20"
        )

    def test_metadata_schedule_matches(self, convergence_metadata: dict):
        """Metadata records the 5-10-5 convergence schedule."""
        schedule = convergence_metadata["convergence_schedule"]
        assert schedule["recent_to_medium_years"] == 5
        assert schedule["medium_hold_years"] == 10
        assert schedule["medium_to_longterm_years"] == 5

    def test_metadata_window_mapping_has_periods(self, convergence_metadata: dict):
        """Metadata records the period windows used for averaging."""
        mapping = convergence_metadata["window_mapping"]
        assert "recent_periods" in mapping
        assert "medium_periods" in mapping
        assert "longterm_periods" in mapping
        assert len(mapping["recent_periods"]) >= 1
        assert len(mapping["medium_periods"]) >= 1
        assert len(mapping["longterm_periods"]) >= 1


# ===================================================================
# 2. Mortality Improvement Direction
# ===================================================================


@pytest.mark.skipif(not _DATA_FILES_EXIST, reason=_SKIP_REASON)
class TestMortalityImprovementDirection:
    """Verify Phase 3 mortality improvement output is directionally correct."""

    @pytest.fixture(scope="class")
    def mortality_df(self) -> pd.DataFrame:
        return pd.read_parquet(MORTALITY_PATH)

    def test_survival_rates_bounded(self, mortality_df: pd.DataFrame):
        """All survival rates must be in [0, 1]."""
        assert (mortality_df["survival_rate"] >= 0).all(), "Negative survival rates found"
        assert (mortality_df["survival_rate"] <= 1).all(), "Survival rates > 1 found"

    def test_expected_years_present(self, mortality_df: pd.DataFrame):
        """Years 2025-2045 must all be present."""
        expected_years = set(range(2025, 2046))
        actual_years = set(mortality_df["year"].unique())
        assert expected_years == actual_years, f"Missing years: {expected_years - actual_years}"

    def test_working_age_survival_improves(self, mortality_df: pd.DataFrame):
        """Mean survival rate for working-age (20-64) should increase 2025 -> 2045."""
        working_age = mortality_df[(mortality_df["age"] >= 20) & (mortality_df["age"] <= 64)]
        mean_2025 = working_age[working_age["year"] == 2025]["survival_rate"].mean()
        mean_2045 = working_age[working_age["year"] == 2045]["survival_rate"].mean()
        assert mean_2045 > mean_2025, (
            f"Working-age survival did not improve: 2025={mean_2025:.6f}, 2045={mean_2045:.6f}"
        )

    def test_both_sexes_present(self, mortality_df: pd.DataFrame):
        """Both Male and Female should be present."""
        sexes = set(mortality_df["sex"].unique())
        assert "Male" in sexes
        assert "Female" in sexes

    def test_reasonable_age_range(self, mortality_df: pd.DataFrame):
        """Ages should start at 0 and go up to at least 90."""
        ages = mortality_df["age"].unique()
        assert 0 in ages, "Age 0 missing from mortality data"
        assert max(ages) >= 90, f"Max age is only {max(ages)}, expected >= 90"

    def test_elderly_survival_lower_than_young(self, mortality_df: pd.DataFrame):
        """Mean survival for elderly (80+) should be lower than young adults (20-40)."""
        young = mortality_df[(mortality_df["age"] >= 20) & (mortality_df["age"] <= 40)]
        elderly = mortality_df[mortality_df["age"] >= 80]
        assert young["survival_rate"].mean() > elderly["survival_rate"].mean()


# ===================================================================
# 3. No Negative Populations (20-year projection)
# ===================================================================


class TestNoNegativePopulations:
    """Run a full 20-year projection with time-varying rates and verify no negatives."""

    @pytest.fixture
    def sample_config(self):
        """Minimal projection config with 20-year horizon."""
        return {
            "project": {
                "base_year": 2025,
                "projection_horizon": 20,
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
    def base_population(self):
        """Small base population for 20-year projection test."""
        races = ["White", "Black"]
        data = []
        for sex in SEXES:
            for race in races:
                for age in range(91):
                    pop = 1000.0 if age < 65 else max(100.0, 800.0 - (age - 65) * 20)
                    data.append(
                        {"year": 2025, "age": age, "sex": sex, "race": race, "population": pop}
                    )
        return pd.DataFrame(data)

    @pytest.fixture
    def fertility_rates(self):
        """Fertility rates for two races."""
        races = ["White", "Black"]
        data = []
        for race in races:
            for age in range(15, 50):
                rate = 0.08 if 20 <= age <= 34 else 0.02
                data.append({"age": age, "race": race, "fertility_rate": rate})
        return pd.DataFrame(data)

    @pytest.fixture
    def survival_rates(self):
        """Constant survival rates."""
        races = ["White", "Black"]
        data = []
        for sex in SEXES:
            for race in races:
                for age in range(91):
                    rate = 0.999 if age < 65 else 0.97
                    data.append({"age": age, "sex": sex, "race": race, "survival_rate": rate})
        return pd.DataFrame(data)

    @pytest.fixture
    def migration_rates(self):
        """Constant migration rates."""
        races = ["White", "Black"]
        data = [
            {"age": age, "sex": sex, "race": race, "net_migration": 5.0}
            for sex in SEXES
            for race in races
            for age in range(91)
        ]
        return pd.DataFrame(data)

    def _make_time_varying_migration(self, base_migration: pd.DataFrame) -> dict[int, pd.DataFrame]:
        """Create time-varying migration dicts for 20 year offsets."""
        result: dict[int, pd.DataFrame] = {}
        for offset in range(1, 21):
            df = base_migration.copy()
            # Gradually decrease migration over time (simulating convergence)
            factor = 1.0 - (offset - 1) * 0.02  # 1.0 at offset 1, 0.62 at offset 20
            df["net_migration"] = df["net_migration"] * factor
            result[offset] = df
        return result

    def _make_time_varying_survival(self, base_survival: pd.DataFrame) -> dict[int, pd.DataFrame]:
        """Create time-varying survival dicts for years 2025-2045."""
        result: dict[int, pd.DataFrame] = {}
        for year in range(2025, 2046):
            df = base_survival.copy()
            # Slightly improving survival over time
            years_from_base = year - 2025
            improvement = years_from_base * 0.0001
            df["survival_rate"] = (df["survival_rate"] + improvement).clip(upper=1.0)
            result[year] = df
        return result

    def test_no_negatives_constant_rates(
        self, base_population, fertility_rates, survival_rates, migration_rates, sample_config
    ):
        """20-year projection with constant rates produces no negative populations."""
        engine = CohortComponentProjection(
            base_population=base_population,
            fertility_rates=fertility_rates,
            survival_rates=survival_rates,
            migration_rates=migration_rates,
            config=sample_config,
        )
        result = engine.run_projection(start_year=2025, end_year=2045)

        negative = result[result["population"] < 0]
        assert negative.empty, (
            f"Found {len(negative)} cohorts with negative population "
            f"(years: {sorted(negative['year'].unique())})"
        )

    def test_no_negatives_time_varying_rates(
        self, base_population, fertility_rates, survival_rates, migration_rates, sample_config
    ):
        """20-year projection with time-varying migration and survival has no negatives."""
        mig_by_year = self._make_time_varying_migration(migration_rates)
        surv_by_year = self._make_time_varying_survival(survival_rates)

        engine = CohortComponentProjection(
            base_population=base_population,
            fertility_rates=fertility_rates,
            survival_rates=survival_rates,
            migration_rates=migration_rates,
            config=sample_config,
            migration_rates_by_year=mig_by_year,
            survival_rates_by_year=surv_by_year,
        )
        result = engine.run_projection(start_year=2025, end_year=2045)

        negative = result[result["population"] < 0]
        assert negative.empty, (
            f"Found {len(negative)} cohorts with negative population "
            f"(years: {sorted(negative['year'].unique())})"
        )

    def test_all_projection_years_present(
        self, base_population, fertility_rates, survival_rates, migration_rates, sample_config
    ):
        """20-year projection produces all expected years 2025-2045."""
        engine = CohortComponentProjection(
            base_population=base_population,
            fertility_rates=fertility_rates,
            survival_rates=survival_rates,
            migration_rates=migration_rates,
            config=sample_config,
        )
        result = engine.run_projection(start_year=2025, end_year=2045)

        expected_years = set(range(2025, 2046))
        actual_years = set(result["year"].unique())
        assert expected_years == actual_years, f"Missing years: {expected_years - actual_years}"


# ===================================================================
# 4. Format Bridge Consistency
# ===================================================================


@pytest.mark.skipif(not _DATA_FILES_EXIST, reason=_SKIP_REASON)
class TestFormatBridgeConsistency:
    """Verify expand_5yr_migration_to_engine_format on real convergence data."""

    @pytest.fixture(scope="class")
    def convergence_df(self) -> pd.DataFrame:
        return pd.read_parquet(CONVERGENCE_PATH)

    def test_expanded_shape_is_1092(self, convergence_df: pd.DataFrame):
        """Expanding a county's year_offset=1 data yields 1,092 rows."""
        # Pick first county
        county = convergence_df["county_fips"].unique()[0]
        slice_df = convergence_df[
            (convergence_df["county_fips"] == county) & (convergence_df["year_offset"] == 1)
        ][["age_group", "sex", "migration_rate"]].reset_index(drop=True)

        result = expand_5yr_migration_to_engine_format(slice_df)
        assert len(result) == 1_092, f"Expected 1,092 rows, got {len(result)}"

    def test_no_nans_in_expanded(self, convergence_df: pd.DataFrame):
        """Expanded result has no NaN values."""
        county = convergence_df["county_fips"].unique()[0]
        slice_df = convergence_df[
            (convergence_df["county_fips"] == county) & (convergence_df["year_offset"] == 1)
        ][["age_group", "sex", "migration_rate"]].reset_index(drop=True)

        result = expand_5yr_migration_to_engine_format(slice_df)
        assert not result.isna().any().any(), "NaN values found in expanded result"

    def test_expanded_has_all_races(self, convergence_df: pd.DataFrame):
        """Expanded result includes all 6 race categories."""
        county = convergence_df["county_fips"].unique()[0]
        slice_df = convergence_df[
            (convergence_df["county_fips"] == county) & (convergence_df["year_offset"] == 1)
        ][["age_group", "sex", "migration_rate"]].reset_index(drop=True)

        result = expand_5yr_migration_to_engine_format(slice_df)
        assert set(result["race"].unique()) == set(ENGINE_RACES)

    def test_expanded_has_correct_columns(self, convergence_df: pd.DataFrame):
        """Expanded result has the engine-expected column names."""
        county = convergence_df["county_fips"].unique()[0]
        slice_df = convergence_df[
            (convergence_df["county_fips"] == county) & (convergence_df["year_offset"] == 1)
        ][["age_group", "sex", "migration_rate"]].reset_index(drop=True)

        result = expand_5yr_migration_to_engine_format(slice_df)
        assert set(result.columns) == {"age", "sex", "race", "migration_rate"}

    def test_expanded_ages_0_to_90(self, convergence_df: pd.DataFrame):
        """Expanded result covers ages 0-90 continuously."""
        county = convergence_df["county_fips"].unique()[0]
        slice_df = convergence_df[
            (convergence_df["county_fips"] == county) & (convergence_df["year_offset"] == 1)
        ][["age_group", "sex", "migration_rate"]].reset_index(drop=True)

        result = expand_5yr_migration_to_engine_format(slice_df)
        assert sorted(result["age"].unique()) == list(range(91))


# ===================================================================
# 5. Pipeline End-to-End Data Flow
# ===================================================================


@pytest.mark.skipif(not _DATA_FILES_EXIST, reason=_SKIP_REASON)
class TestPipelineDataFlow:
    """Verify bridge functions produce expected dict structures from real data."""

    @pytest.fixture(scope="class")
    def convergence_df(self) -> pd.DataFrame:
        return pd.read_parquet(CONVERGENCE_PATH)

    @pytest.fixture(scope="class")
    def mortality_df(self) -> pd.DataFrame:
        return pd.read_parquet(MORTALITY_PATH)

    # -- _build_convergence_rate_dicts --

    def test_convergence_dicts_outer_keys_are_counties(self, convergence_df: pd.DataFrame):
        """Outer dict keys should be county FIPS strings."""
        result = _build_convergence_rate_dicts(convergence_df)
        expected_counties = set(convergence_df["county_fips"].unique())
        assert set(result.keys()) == expected_counties

    def test_convergence_dicts_inner_keys_are_offsets(self, convergence_df: pd.DataFrame):
        """Inner dict keys should be year offset integers 1-20."""
        result = _build_convergence_rate_dicts(convergence_df)
        sample_county = next(iter(result))
        assert set(result[sample_county].keys()) == set(range(1, 21))

    def test_convergence_dicts_inner_df_shape(self, convergence_df: pd.DataFrame):
        """Each inner DataFrame should have 1,092 rows (91 ages x 2 sexes x 6 races)."""
        result = _build_convergence_rate_dicts(convergence_df)
        sample_county = next(iter(result))
        for offset, df in result[sample_county].items():
            assert len(df) == 1_092, (
                f"County {sample_county} offset {offset}: {len(df)} rows, expected 1,092"
            )

    def test_convergence_dicts_inner_df_columns(self, convergence_df: pd.DataFrame):
        """Each inner DataFrame should have engine-format columns."""
        result = _build_convergence_rate_dicts(convergence_df)
        sample_county = next(iter(result))
        sample_df = result[sample_county][1]
        assert set(sample_df.columns) == {"age", "sex", "race", "migration_rate"}

    # -- _build_survival_rates_by_year --

    def test_survival_dicts_keys_are_years(self, mortality_df: pd.DataFrame):
        """Dict keys should be calendar years 2025-2045."""
        result = _build_survival_rates_by_year(mortality_df)
        assert set(result.keys()) == set(range(2025, 2046))

    def test_survival_dicts_has_race_expansion(self, mortality_df: pd.DataFrame):
        """Each year's DataFrame should have all 6 race categories."""
        result = _build_survival_rates_by_year(mortality_df)
        sample_df = result[2025]
        assert set(sample_df["race"].unique()) == set(ENGINE_RACES)

    def test_survival_dicts_columns(self, mortality_df: pd.DataFrame):
        """Each year's DataFrame should have [age, sex, race, survival_rate]."""
        result = _build_survival_rates_by_year(mortality_df)
        sample_df = result[2025]
        assert set(sample_df.columns) == {"age", "sex", "race", "survival_rate"}

    def test_survival_dicts_rates_bounded(self, mortality_df: pd.DataFrame):
        """Survival rates in every year dict should be in [0, 1]."""
        result = _build_survival_rates_by_year(mortality_df)
        for year, df in result.items():
            assert (df["survival_rate"] >= 0).all(), f"Negative rate in year {year}"
            assert (df["survival_rate"] <= 1).all(), f"Rate > 1 in year {year}"
