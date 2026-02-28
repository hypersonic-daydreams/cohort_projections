"""
Tests for pipeline orchestrator wiring correctness.

Validates that run_residual_migration_pipeline and run_convergence_pipeline
correctly wire together individually-tested components. Uses small synthetic
data and mocks file I/O while keeping computation logic real.

Coverage target: PP4-02 (pipeline orchestrator tests).
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from cohort_projections.data.load.census_age_sex_population import AGE_GROUP_LABELS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEX_CATEGORIES = ["Male", "Female"]
CELLS_PER_COUNTY = len(AGE_GROUP_LABELS) * len(SEX_CATEGORIES)  # 36

# Use 3 synthetic counties to keep tests fast
SYNTHETIC_COUNTIES = ["38001", "38003", "38005"]

# The 5 historical periods used in residual migration
PERIODS = [
    (2000, 2005),
    (2005, 2010),
    (2010, 2015),
    (2015, 2020),
    (2020, 2024),
]


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _make_population_snapshot(
    counties: list[str],
    base_pop: float = 500.0,
) -> pd.DataFrame:
    """Create a synthetic population DataFrame for multiple counties.

    Each county gets all 18 age groups x 2 sexes with a stable base
    population that varies by age (younger ages higher).
    """
    records = []
    for county in counties:
        for i, ag in enumerate(AGE_GROUP_LABELS):
            for sex in SEX_CATEGORIES:
                # Population decreases with age to be demographically plausible
                pop = max(50.0, base_pop - i * 20)
                records.append(
                    {
                        "county_fips": county,
                        "age_group": ag,
                        "sex": sex,
                        "population": pop,
                    }
                )
    return pd.DataFrame(records)


def _make_survival_rates() -> pd.DataFrame:
    """Create synthetic survival rates for all 18 age groups x 2 sexes."""
    records = []
    for ag in AGE_GROUP_LABELS:
        for sex in SEX_CATEGORIES:
            idx = AGE_GROUP_LABELS.index(ag)
            rate = max(0.5, 1.0 - idx * 0.02)
            if ag == "85+":
                rate = 0.60
            records.append(
                {"age_group": ag, "sex": sex, "survival_rate_5yr": rate}
            )
    return pd.DataFrame(records)


def _make_period_rates(
    counties: list[str],
    periods: list[tuple[int, int]] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic Phase 1 period rates.

    Produces deterministic rates in the range [-0.05, +0.05] which is
    demographically plausible for annual migration rates.
    """
    if periods is None:
        periods = PERIODS

    rng = np.random.default_rng(seed)
    records = []
    for county in counties:
        for ps, pe in periods:
            for ag in AGE_GROUP_LABELS:
                for sex in SEX_CATEGORIES:
                    base = hash((county, ag, sex)) % 100 / 1000.0 - 0.05
                    period_shift = (ps - 2000) * 0.002
                    rate = base + period_shift + rng.normal(0, 0.005)
                    records.append(
                        {
                            "county_fips": county,
                            "age_group": ag,
                            "sex": sex,
                            "period_start": ps,
                            "period_end": pe,
                            "pop_start": 1000.0,
                            "expected_pop": 980.0,
                            "pop_end": 1010.0,
                            "net_migration": rate * 1000,
                            "migration_rate": rate,
                        }
                    )
    return pd.DataFrame(records)


def _make_gq_historical(
    counties: list[str],
    years: list[int],
) -> pd.DataFrame:
    """Create synthetic GQ historical data with small constant GQ populations."""
    records = []
    for county in counties:
        for year in years:
            for ag in AGE_GROUP_LABELS:
                for sex in SEX_CATEGORIES:
                    # Small GQ population (10% of typical population)
                    records.append(
                        {
                            "county_fips": county,
                            "year": year,
                            "age_group": ag,
                            "sex": sex,
                            "gq_population": 5.0,
                        }
                    )
    return pd.DataFrame(records)


def _make_pep_data(
    counties: list[str],
    year_range: tuple[int, int] = (2001, 2024),
) -> pd.DataFrame:
    """Create synthetic PEP net migration data for recalibration."""
    records = []
    for county in counties:
        for year in range(year_range[0], year_range[1] + 1):
            records.append(
                {
                    "geoid": county,
                    "year": year,
                    "netmig": 50.0,  # positive net migration
                }
            )
    return pd.DataFrame(records)


def _make_minimal_config(
    enable_gq: bool = False,
    enable_pep_recal: bool = False,
    enable_college: bool = False,
    enable_dampening: bool = False,
    enable_male_dampening: bool = False,
) -> dict:
    """Build a minimal config dict matching the structure expected by pipelines.

    Disables most optional features by default to isolate pipeline wiring.
    """
    return {
        "project": {
            "base_year": 2025,
            "projection_horizon": 20,
        },
        "rates": {
            "migration": {
                "domestic": {
                    "residual": {
                        "periods": [
                            [2000, 2005],
                            [2005, 2010],
                            [2010, 2015],
                            [2015, 2020],
                            [2020, 2024],
                        ],
                        "averaging": "simple_average",
                        "gq_correction": {
                            "enabled": enable_gq,
                            "historical_gq_path": "data/processed/gq_historical.parquet",
                        },
                        "pep_recalibration": {
                            "enabled": enable_pep_recal,
                            "counties": ["38005"],
                            "pep_data_path": "data/processed/pep_components.parquet",
                            "near_zero_threshold": 10,
                        },
                    },
                    "dampening": {
                        "enabled": enable_dampening,
                        "factor": 0.60,
                        "counties": ["38105"],
                        "boom_periods": [[2005, 2010], [2010, 2015]],
                    },
                    "adjustments": {
                        "college_age": {
                            "enabled": enable_college,
                            "method": "smooth",
                            "counties": ["38001"],
                            "age_groups": ["15-19", "20-24"],
                            "blend_factor": 0.5,
                        },
                        "male_dampening": {
                            "enabled": enable_male_dampening,
                            "factor": 0.80,
                            "boom_periods": [[2005, 2010], [2010, 2015]],
                        },
                    },
                },
                "interpolation": {
                    "recent_period": [2022, 2024],
                    "medium_period": [2014, 2024],
                    "longterm_period": [2000, 2024],
                    "convergence_schedule": {
                        "recent_to_medium_years": 5,
                        "medium_hold_years": 10,
                        "medium_to_longterm_years": 5,
                    },
                    "rate_cap": {
                        "enabled": True,
                        "college_ages": ["15-19", "20-24"],
                        "college_cap": 0.15,
                        "general_cap": 0.08,
                    },
                },
            },
        },
        "scenarios": {
            "high_growth": {
                "migration_floor": {
                    "enabled": False,
                    "floor_value": 0.0,
                },
            },
        },
        "output": {
            "compression": "gzip",
        },
    }


# ===========================================================================
# Tests: run_residual_migration_pipeline
# ===========================================================================


class TestResidualMigrationPipelineWiring:
    """Tests for run_residual_migration_pipeline orchestration.

    These tests mock file I/O (population loading, survival rate loading,
    GQ data reading, parquet/JSON writing) while exercising the real
    computation logic to verify correct wiring.
    """

    @pytest.fixture
    def synthetic_populations(self) -> dict[int, pd.DataFrame]:
        """Population snapshots for all 6 time points."""
        pops = {}
        for year in [2000, 2005, 2010, 2015, 2020, 2024]:
            # Slight growth over time to produce plausible residual rates
            growth = 1.0 + (year - 2000) * 0.005
            pops[year] = _make_population_snapshot(
                SYNTHETIC_COUNTIES, base_pop=500.0 * growth
            )
        return pops

    @pytest.fixture
    def synthetic_survival(self) -> pd.DataFrame:
        """Survival rates for all age-sex cells."""
        return _make_survival_rates()

    def _run_pipeline_with_mocks(
        self,
        config: dict,
        populations: dict[int, pd.DataFrame],
        survival_rates: pd.DataFrame,
        gq_historical: pd.DataFrame | None = None,
        pep_data: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Run pipeline with all I/O mocked out."""
        from cohort_projections.data.process.residual_migration import (
            run_residual_migration_pipeline,
        )

        with (
            patch(
                "cohort_projections.data.process.residual_migration.assemble_period_populations",
                return_value=populations,
            ),
            patch(
                "cohort_projections.data.process.residual_migration._load_survival_rates",
                return_value=survival_rates,
            ),
            patch(
                "cohort_projections.data.process.residual_migration.pd.read_parquet",
                side_effect=lambda path: (
                    gq_historical
                    if "gq" in str(path)
                    else (pep_data if pep_data is not None else pd.DataFrame())
                ),
            ),
            patch("cohort_projections.data.process.residual_migration.Path.exists", return_value=True),
            patch("cohort_projections.data.process.residual_migration.Path.mkdir"),
            patch("pandas.DataFrame.to_parquet"),
            patch("builtins.open", mock_open()),
            patch("json.dump"),
        ):
            return run_residual_migration_pipeline(config=config)

    def test_pipeline_produces_all_periods_and_averaged(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """Pipeline returns both 'all_periods' and 'averaged' DataFrames."""
        config = _make_minimal_config()
        result = self._run_pipeline_with_mocks(
            config, synthetic_populations, synthetic_survival
        )

        assert "all_periods" in result
        assert "averaged" in result
        assert isinstance(result["all_periods"], pd.DataFrame)
        assert isinstance(result["averaged"], pd.DataFrame)

    def test_all_periods_has_expected_columns(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """All-periods output contains the residual migration schema columns."""
        config = _make_minimal_config()
        result = self._run_pipeline_with_mocks(
            config, synthetic_populations, synthetic_survival
        )

        expected_cols = {
            "county_fips",
            "age_group",
            "sex",
            "period_start",
            "period_end",
            "pop_start",
            "expected_pop",
            "pop_end",
            "net_migration",
            "migration_rate",
        }
        actual_cols = set(result["all_periods"].columns)
        assert expected_cols.issubset(actual_cols)

    def test_averaged_has_expected_columns(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """Averaged output contains group columns plus rate and count."""
        config = _make_minimal_config()
        result = self._run_pipeline_with_mocks(
            config, synthetic_populations, synthetic_survival
        )

        expected_cols = {
            "county_fips",
            "age_group",
            "sex",
            "migration_rate",
            "net_migration",
            "n_periods",
        }
        actual_cols = set(result["averaged"].columns)
        assert expected_cols == actual_cols

    def test_all_periods_covers_five_periods(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """All-periods output contains data for all 5 historical periods."""
        config = _make_minimal_config()
        result = self._run_pipeline_with_mocks(
            config, synthetic_populations, synthetic_survival
        )

        all_periods = result["all_periods"]
        period_pairs = (
            all_periods[["period_start", "period_end"]]
            .drop_duplicates()
            .apply(lambda r: (int(r["period_start"]), int(r["period_end"])), axis=1)
            .tolist()
        )
        assert sorted(period_pairs) == sorted(PERIODS)

    def test_averaged_has_one_row_per_cell(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """Averaged output has exactly one row per county x age_group x sex."""
        config = _make_minimal_config()
        result = self._run_pipeline_with_mocks(
            config, synthetic_populations, synthetic_survival
        )

        averaged = result["averaged"]
        n_counties = len(SYNTHETIC_COUNTIES)
        expected_rows = n_counties * CELLS_PER_COUNTY
        assert len(averaged) == expected_rows

    def test_averaged_n_periods_equals_five(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """Each averaged cell reports n_periods=5 (all periods contributed)."""
        config = _make_minimal_config()
        result = self._run_pipeline_with_mocks(
            config, synthetic_populations, synthetic_survival
        )

        assert (result["averaged"]["n_periods"] == 5).all()

    def test_no_nan_in_migration_rates(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """No NaN values leak through the pipeline into migration rates."""
        config = _make_minimal_config()
        result = self._run_pipeline_with_mocks(
            config, synthetic_populations, synthetic_survival
        )

        assert not result["all_periods"]["migration_rate"].isna().any()
        assert not result["averaged"]["migration_rate"].isna().any()

    def test_migration_rates_within_plausible_bounds(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """Migration rates are within demographically plausible bounds [-1, +1]."""
        config = _make_minimal_config()
        result = self._run_pipeline_with_mocks(
            config, synthetic_populations, synthetic_survival
        )

        rates = result["averaged"]["migration_rate"]
        assert (rates >= -1.0).all(), "Migration rate below -1.0 (impossible)"
        assert (rates <= 1.0).all(), "Migration rate above +1.0 (implausible)"

    def test_all_counties_present_in_output(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """All input counties appear in the output."""
        config = _make_minimal_config()
        result = self._run_pipeline_with_mocks(
            config, synthetic_populations, synthetic_survival
        )

        output_counties = sorted(result["averaged"]["county_fips"].unique())
        assert output_counties == sorted(SYNTHETIC_COUNTIES)

    def test_all_age_groups_present_in_output(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """All 18 age groups appear in the output."""
        config = _make_minimal_config()
        result = self._run_pipeline_with_mocks(
            config, synthetic_populations, synthetic_survival
        )

        output_ags = sorted(
            result["averaged"]["age_group"].unique(),
            key=lambda x: AGE_GROUP_LABELS.index(x),
        )
        assert output_ags == AGE_GROUP_LABELS

    def test_both_sexes_present(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """Both Male and Female appear in the output."""
        config = _make_minimal_config()
        result = self._run_pipeline_with_mocks(
            config, synthetic_populations, synthetic_survival
        )

        assert sorted(result["averaged"]["sex"].unique()) == ["Female", "Male"]

    def test_pipeline_with_gq_correction_enabled(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """Pipeline completes when GQ correction is enabled."""
        config = _make_minimal_config(enable_gq=True)
        gq_data = _make_gq_historical(
            SYNTHETIC_COUNTIES,
            years=[2000, 2005, 2010, 2015, 2020, 2024],
        )

        result = self._run_pipeline_with_mocks(
            config, synthetic_populations, synthetic_survival,
            gq_historical=gq_data,
        )

        assert len(result["averaged"]) == len(SYNTHETIC_COUNTIES) * CELLS_PER_COUNTY

    def test_gq_correction_reduces_effective_population(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """GQ-corrected pipeline uses lower effective populations."""
        from cohort_projections.data.process.residual_migration import (
            subtract_gq_from_populations,
        )

        gq_data = _make_gq_historical(
            SYNTHETIC_COUNTIES,
            years=[2000, 2005, 2010, 2015, 2020, 2024],
        )

        corrected = subtract_gq_from_populations(
            synthetic_populations, gq_data
        )

        for year in synthetic_populations:
            orig_total = synthetic_populations[year]["population"].sum()
            corrected_total = corrected[year]["population"].sum()
            assert corrected_total < orig_total

    def test_pipeline_with_college_smoothing_enabled(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """Pipeline completes when college-age smoothing is enabled."""
        config = _make_minimal_config(enable_college=True)
        result = self._run_pipeline_with_mocks(
            config, synthetic_populations, synthetic_survival
        )

        # Verify output is well-formed
        assert not result["averaged"]["migration_rate"].isna().any()
        assert len(result["averaged"]) > 0

    def test_pipeline_with_dampening_enabled(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """Pipeline completes when oil-boom dampening is enabled."""
        # Add an oil county to synthetic data
        counties_with_oil = SYNTHETIC_COUNTIES + ["38105"]
        pops = {}
        for year in [2000, 2005, 2010, 2015, 2020, 2024]:
            growth = 1.0 + (year - 2000) * 0.005
            pops[year] = _make_population_snapshot(
                counties_with_oil, base_pop=500.0 * growth
            )

        config = _make_minimal_config(enable_dampening=True)
        result = self._run_pipeline_with_mocks(
            config, pops, synthetic_survival
        )

        assert "38105" in result["averaged"]["county_fips"].values

    def test_pipeline_with_all_features_enabled(
        self,
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """Pipeline completes with all optional features enabled simultaneously."""
        counties = SYNTHETIC_COUNTIES + ["38005", "38105"]
        # Remove duplicates
        counties = sorted(set(counties))

        pops = {}
        for year in [2000, 2005, 2010, 2015, 2020, 2024]:
            growth = 1.0 + (year - 2000) * 0.005
            pops[year] = _make_population_snapshot(counties, base_pop=500.0 * growth)

        gq_data = _make_gq_historical(counties, [2000, 2005, 2010, 2015, 2020, 2024])
        pep_data = _make_pep_data(["38005"])

        config = _make_minimal_config(
            enable_gq=True,
            enable_pep_recal=True,
            enable_college=True,
            enable_dampening=True,
            enable_male_dampening=True,
        )

        result = self._run_pipeline_with_mocks(
            config, pops, synthetic_survival,
            gq_historical=gq_data,
            pep_data=pep_data,
        )

        assert not result["averaged"]["migration_rate"].isna().any()
        assert result["averaged"]["n_periods"].iloc[0] == 5

    def test_period_rows_per_county_is_36(
        self,
        synthetic_populations: dict[int, pd.DataFrame],
        synthetic_survival: pd.DataFrame,
    ) -> None:
        """Each period for each county has exactly 36 rows (18 ages x 2 sexes)."""
        config = _make_minimal_config()
        result = self._run_pipeline_with_mocks(
            config, synthetic_populations, synthetic_survival
        )

        grouped = result["all_periods"].groupby(
            ["county_fips", "period_start", "period_end"]
        ).size()
        assert (grouped == CELLS_PER_COUNTY).all()


# ===========================================================================
# Tests: run_convergence_pipeline
# ===========================================================================


class TestConvergencePipelineWiring:
    """Tests for run_convergence_pipeline orchestration.

    Mocks file I/O (parquet reads and writes) while exercising the real
    convergence interpolation computation.
    """

    @pytest.fixture
    def synthetic_phase1_rates(self) -> pd.DataFrame:
        """Synthetic Phase 1 period rates for 3 counties."""
        return _make_period_rates(SYNTHETIC_COUNTIES)

    def _run_convergence_with_mocks(
        self,
        config: dict,
        phase1_rates: pd.DataFrame,
        variant: str | None = None,
    ) -> dict:
        """Run convergence pipeline with file I/O mocked out."""
        from cohort_projections.data.process.convergence_interpolation import (
            run_convergence_pipeline,
        )

        with (
            patch(
                "cohort_projections.data.process.convergence_interpolation.pd.read_parquet",
                return_value=phase1_rates,
            ),
            patch(
                "cohort_projections.data.process.convergence_interpolation.Path.mkdir",
            ),
            patch("pandas.DataFrame.to_parquet"),
            patch("builtins.open", mock_open()),
            patch("json.dump"),
        ):
            return run_convergence_pipeline(config=config, variant=variant)

    def test_pipeline_returns_expected_keys(
        self,
        synthetic_phase1_rates: pd.DataFrame,
    ) -> None:
        """Pipeline returns dict with rates_by_year, output_path, metadata_path, total_rows."""
        config = _make_minimal_config()
        result = self._run_convergence_with_mocks(config, synthetic_phase1_rates)

        assert "rates_by_year" in result
        assert "output_path" in result
        assert "metadata_path" in result
        assert "total_rows" in result

    def test_rates_by_year_covers_full_horizon(
        self,
        synthetic_phase1_rates: pd.DataFrame,
    ) -> None:
        """Rates dict has entries for years 1 through projection_horizon."""
        config = _make_minimal_config()
        result = self._run_convergence_with_mocks(config, synthetic_phase1_rates)

        horizon = config["project"]["projection_horizon"]
        assert len(result["rates_by_year"]) == horizon
        assert sorted(result["rates_by_year"].keys()) == list(range(1, horizon + 1))

    def test_each_year_has_expected_columns(
        self,
        synthetic_phase1_rates: pd.DataFrame,
    ) -> None:
        """Each year DataFrame has the correct schema columns."""
        config = _make_minimal_config()
        result = self._run_convergence_with_mocks(config, synthetic_phase1_rates)

        expected_cols = {"county_fips", "age_group", "sex", "migration_rate"}
        for year_offset, year_df in result["rates_by_year"].items():
            assert set(year_df.columns) == expected_cols, (
                f"Year {year_offset} has wrong columns: {set(year_df.columns)}"
            )

    def test_each_year_has_correct_row_count(
        self,
        synthetic_phase1_rates: pd.DataFrame,
    ) -> None:
        """Each year has n_counties x 36 rows."""
        config = _make_minimal_config()
        result = self._run_convergence_with_mocks(config, synthetic_phase1_rates)

        n_counties = len(SYNTHETIC_COUNTIES)
        expected_per_year = n_counties * CELLS_PER_COUNTY

        for year_offset, year_df in result["rates_by_year"].items():
            assert len(year_df) == expected_per_year, (
                f"Year {year_offset}: expected {expected_per_year}, got {len(year_df)}"
            )

    def test_total_rows_matches_years_times_cells(
        self,
        synthetic_phase1_rates: pd.DataFrame,
    ) -> None:
        """total_rows equals projection_horizon x n_counties x 36."""
        config = _make_minimal_config()
        result = self._run_convergence_with_mocks(config, synthetic_phase1_rates)

        horizon = config["project"]["projection_horizon"]
        n_counties = len(SYNTHETIC_COUNTIES)
        expected_total = horizon * n_counties * CELLS_PER_COUNTY
        assert result["total_rows"] == expected_total

    def test_no_nan_in_convergence_rates(
        self,
        synthetic_phase1_rates: pd.DataFrame,
    ) -> None:
        """No NaN values in any year's migration rates."""
        config = _make_minimal_config()
        result = self._run_convergence_with_mocks(config, synthetic_phase1_rates)

        for year_offset, year_df in result["rates_by_year"].items():
            assert not year_df["migration_rate"].isna().any(), (
                f"Year {year_offset} has NaN migration rates"
            )

    def test_rates_within_cap_bounds(
        self,
        synthetic_phase1_rates: pd.DataFrame,
    ) -> None:
        """When rate cap is enabled, rates stay within configured bounds."""
        config = _make_minimal_config()
        rate_cap = config["rates"]["migration"]["interpolation"]["rate_cap"]
        general_cap = rate_cap["general_cap"]
        college_cap = rate_cap["college_cap"]
        college_ages = rate_cap["college_ages"]

        result = self._run_convergence_with_mocks(config, synthetic_phase1_rates)

        for year_offset, year_df in result["rates_by_year"].items():
            # Check non-college ages
            non_college = year_df[~year_df["age_group"].isin(college_ages)]
            if len(non_college) > 0:
                assert non_college["migration_rate"].max() <= general_cap + 1e-10, (
                    f"Year {year_offset}: non-college rate exceeds general cap"
                )
                assert non_college["migration_rate"].min() >= -general_cap - 1e-10, (
                    f"Year {year_offset}: non-college rate below negative general cap"
                )

            # Check college ages
            college = year_df[year_df["age_group"].isin(college_ages)]
            if len(college) > 0:
                assert college["migration_rate"].max() <= college_cap + 1e-10, (
                    f"Year {year_offset}: college rate exceeds college cap"
                )
                assert college["migration_rate"].min() >= -college_cap - 1e-10, (
                    f"Year {year_offset}: college rate below negative college cap"
                )

    def test_year5_equals_medium_window_average(
        self,
        synthetic_phase1_rates: pd.DataFrame,
    ) -> None:
        """Year 5 rates equal the medium window average (end of phase 1)."""
        config = _make_minimal_config()
        # Disable rate cap so convergence math is exact
        config["rates"]["migration"]["interpolation"]["rate_cap"] = {"enabled": False}

        result = self._run_convergence_with_mocks(config, synthetic_phase1_rates)

        # Manually compute the medium window average from input data
        from cohort_projections.data.process.convergence_interpolation import (
            _map_config_window_to_periods,
            compute_period_window_averages,
        )

        available_periods = sorted(
            synthetic_phase1_rates[["period_start", "period_end"]]
            .drop_duplicates()
            .apply(lambda r: (int(r["period_start"]), int(r["period_end"])), axis=1)
            .tolist()
        )
        medium_range = config["rates"]["migration"]["interpolation"]["medium_period"]
        medium_periods = _map_config_window_to_periods(medium_range, available_periods)

        _, medium_rates, _ = compute_period_window_averages(
            synthetic_phase1_rates,
            recent_periods=_map_config_window_to_periods(
                config["rates"]["migration"]["interpolation"]["recent_period"],
                available_periods,
            ),
            medium_periods=medium_periods,
            longterm_periods=_map_config_window_to_periods(
                config["rates"]["migration"]["interpolation"]["longterm_period"],
                available_periods,
            ),
        )

        year5 = (
            result["rates_by_year"][5]
            .sort_values(["county_fips", "age_group", "sex"])
            .reset_index(drop=True)
        )
        medium_sorted = (
            medium_rates
            .sort_values(["county_fips", "age_group", "sex"])
            .reset_index(drop=True)
        )

        np.testing.assert_allclose(
            year5["migration_rate"].values,
            medium_sorted["migration_rate"].values,
            atol=1e-10,
        )

    def test_year20_equals_longterm_window_average(
        self,
        synthetic_phase1_rates: pd.DataFrame,
    ) -> None:
        """Year 20 rates equal the long-term window average (end of phase 3)."""
        config = _make_minimal_config()
        config["rates"]["migration"]["interpolation"]["rate_cap"] = {"enabled": False}

        result = self._run_convergence_with_mocks(config, synthetic_phase1_rates)

        from cohort_projections.data.process.convergence_interpolation import (
            _map_config_window_to_periods,
            compute_period_window_averages,
        )

        available_periods = sorted(
            synthetic_phase1_rates[["period_start", "period_end"]]
            .drop_duplicates()
            .apply(lambda r: (int(r["period_start"]), int(r["period_end"])), axis=1)
            .tolist()
        )

        longterm_range = config["rates"]["migration"]["interpolation"]["longterm_period"]
        longterm_periods = _map_config_window_to_periods(longterm_range, available_periods)

        _, _, longterm_rates = compute_period_window_averages(
            synthetic_phase1_rates,
            recent_periods=_map_config_window_to_periods(
                config["rates"]["migration"]["interpolation"]["recent_period"],
                available_periods,
            ),
            medium_periods=_map_config_window_to_periods(
                config["rates"]["migration"]["interpolation"]["medium_period"],
                available_periods,
            ),
            longterm_periods=longterm_periods,
        )

        year20 = (
            result["rates_by_year"][20]
            .sort_values(["county_fips", "age_group", "sex"])
            .reset_index(drop=True)
        )
        longterm_sorted = (
            longterm_rates
            .sort_values(["county_fips", "age_group", "sex"])
            .reset_index(drop=True)
        )

        np.testing.assert_allclose(
            year20["migration_rate"].values,
            longterm_sorted["migration_rate"].values,
            atol=1e-10,
        )

    def test_convergence_schedule_respected(
        self,
        synthetic_phase1_rates: pd.DataFrame,
    ) -> None:
        """Years 6-15 hold at medium rate (phase 2 of 5-10-5 schedule)."""
        config = _make_minimal_config()
        config["rates"]["migration"]["interpolation"]["rate_cap"] = {"enabled": False}

        result = self._run_convergence_with_mocks(config, synthetic_phase1_rates)

        # All years 6-15 should have identical rates (medium hold)
        year6_rates = result["rates_by_year"][6]["migration_rate"].values
        for year in range(7, 16):
            np.testing.assert_allclose(
                result["rates_by_year"][year]["migration_rate"].values,
                year6_rates,
                atol=1e-10,
                err_msg=f"Year {year} should equal year 6 (medium hold phase)",
            )

    def test_all_counties_present_in_every_year(
        self,
        synthetic_phase1_rates: pd.DataFrame,
    ) -> None:
        """All input counties appear in each year of the output."""
        config = _make_minimal_config()
        result = self._run_convergence_with_mocks(config, synthetic_phase1_rates)

        for year_offset, year_df in result["rates_by_year"].items():
            output_counties = sorted(year_df["county_fips"].unique())
            assert output_counties == sorted(SYNTHETIC_COUNTIES), (
                f"Year {year_offset}: missing counties"
            )

    def test_rate_cap_disabled_allows_higher_rates(
        self,
    ) -> None:
        """With rate cap disabled, rates can exceed the default cap thresholds."""
        # Create data with extreme rates that would normally be capped
        counties = ["38001"]
        records = []
        for ps, pe in PERIODS:
            for ag in AGE_GROUP_LABELS:
                for sex in SEX_CATEGORIES:
                    # Use a rate of 0.12 for non-college ages (above 0.08 cap)
                    records.append(
                        {
                            "county_fips": "38001",
                            "age_group": ag,
                            "sex": sex,
                            "period_start": ps,
                            "period_end": pe,
                            "pop_start": 1000.0,
                            "expected_pop": 980.0,
                            "pop_end": 1010.0,
                            "net_migration": 120.0,
                            "migration_rate": 0.12,
                        }
                    )

        phase1_rates = pd.DataFrame(records)
        config = _make_minimal_config()
        config["rates"]["migration"]["interpolation"]["rate_cap"] = {"enabled": False}

        result = self._run_convergence_with_mocks(config, phase1_rates)

        # With cap disabled, non-college age rates should remain at 0.12
        year10 = result["rates_by_year"][10]
        non_college = year10[~year10["age_group"].isin(["15-19", "20-24"])]
        assert non_college["migration_rate"].max() > 0.08

    def test_convergence_with_custom_horizon(
        self,
        synthetic_phase1_rates: pd.DataFrame,
    ) -> None:
        """Custom projection horizon (e.g., 10 years) produces correct number of years."""
        config = _make_minimal_config()
        config["project"]["projection_horizon"] = 10

        result = self._run_convergence_with_mocks(config, synthetic_phase1_rates)

        assert len(result["rates_by_year"]) == 10
        assert sorted(result["rates_by_year"].keys()) == list(range(1, 11))

    def test_output_path_is_path_object(
        self,
        synthetic_phase1_rates: pd.DataFrame,
    ) -> None:
        """Output paths are Path objects."""
        config = _make_minimal_config()
        result = self._run_convergence_with_mocks(config, synthetic_phase1_rates)

        assert isinstance(result["output_path"], Path)
        assert isinstance(result["metadata_path"], Path)

    def test_baseline_variant_uses_default_filename(
        self,
        synthetic_phase1_rates: pd.DataFrame,
    ) -> None:
        """Baseline (variant=None) saves to convergence_rates_by_year.parquet."""
        config = _make_minimal_config()
        result = self._run_convergence_with_mocks(
            config, synthetic_phase1_rates, variant=None
        )

        assert result["output_path"].name == "convergence_rates_by_year.parquet"

    def test_high_variant_uses_suffixed_filename(
        self,
        synthetic_phase1_rates: pd.DataFrame,
    ) -> None:
        """High variant saves to convergence_rates_by_year_high.parquet.

        Note: We don't actually exercise the BEBR file loading for the high
        variant (that would require additional mocks for the BEBR parquet files).
        Instead, we verify just the filename convention by checking the path
        is constructed correctly from the variant parameter. The actual high
        variant computation is tested via _lift_window_averages unit tests.
        """
        from cohort_projections.data.process.convergence_interpolation import (
            run_convergence_pipeline,
        )

        config = _make_minimal_config()

        # For the high variant, the pipeline tries to read BEBR files.
        # We mock all I/O including the BEBR-specific reads.
        bebr_baseline = pd.DataFrame(
            {
                "county_fips": ["38001"] * 2,
                "net_migration": [100.0, 200.0],
            }
        )
        bebr_high = pd.DataFrame(
            {
                "county_fips": ["38001"] * 2,
                "net_migration": [150.0, 250.0],
            }
        )

        # The pipeline reads multiple parquet files; we return appropriate
        # data based on the filename pattern.
        def _mock_read_parquet(path):
            path_str = str(path)
            if "residual_migration_rates" in path_str and "averaged" not in path_str:
                return synthetic_phase1_rates
            elif "pep_baseline" in path_str:
                return bebr_baseline
            elif "pep_high" in path_str:
                return bebr_high
            return synthetic_phase1_rates

        with (
            patch(
                "cohort_projections.data.process.convergence_interpolation.pd.read_parquet",
                side_effect=_mock_read_parquet,
            ),
            patch(
                "cohort_projections.data.process.convergence_interpolation.Path.mkdir",
            ),
            patch("pandas.DataFrame.to_parquet"),
            patch("builtins.open", mock_open()),
            patch("json.dump"),
        ):
            result = run_convergence_pipeline(config=config, variant="high")

        assert result["output_path"].name == "convergence_rates_by_year_high.parquet"
        assert result["metadata_path"].name == "convergence_metadata_high.json"


# ===========================================================================
# Tests: Cross-pipeline data flow
# ===========================================================================


class TestCrossPipelineDataFlow:
    """Tests verifying that residual pipeline output is compatible with
    convergence pipeline input.

    These tests ensure the two pipelines can be wired together correctly
    by verifying the schema contract between them.
    """

    def test_residual_output_schema_matches_convergence_input(self) -> None:
        """Residual all_periods output has the columns convergence pipeline expects."""
        # The convergence pipeline reads a parquet with these columns:
        convergence_required_cols = {
            "county_fips",
            "age_group",
            "sex",
            "period_start",
            "period_end",
            "migration_rate",
        }

        # Build synthetic residual pipeline output
        from cohort_projections.data.process.residual_migration import (
            compute_residual_migration_rates,
        )

        pop = _make_population_snapshot(["38001"])
        surv = _make_survival_rates()
        result = compute_residual_migration_rates(
            pop_start=pop, pop_end=pop, survival_rates=surv, period=(2020, 2024)
        )

        assert convergence_required_cols.issubset(set(result.columns))

    def test_period_rates_have_consistent_dimensions(self) -> None:
        """Each period produces the same number of cells per county."""
        from cohort_projections.data.process.residual_migration import (
            compute_residual_migration_rates,
        )

        pop = _make_population_snapshot(["38001", "38003"])
        surv = _make_survival_rates()

        period1 = compute_residual_migration_rates(
            pop_start=pop, pop_end=pop, survival_rates=surv, period=(2000, 2005)
        )
        period2 = compute_residual_migration_rates(
            pop_start=pop, pop_end=pop, survival_rates=surv, period=(2020, 2024)
        )

        cells_per_county_p1 = len(period1) / period1["county_fips"].nunique()
        cells_per_county_p2 = len(period2) / period2["county_fips"].nunique()

        assert cells_per_county_p1 == cells_per_county_p2
        assert cells_per_county_p1 == CELLS_PER_COUNTY

    def test_convergence_input_has_all_required_period_columns(self) -> None:
        """Period rates from residual pipeline have period_start and period_end.

        The convergence pipeline uses these to determine window mappings.
        """
        from cohort_projections.data.process.residual_migration import (
            compute_residual_migration_rates,
        )

        pop = _make_population_snapshot(["38001"])
        surv = _make_survival_rates()

        result = compute_residual_migration_rates(
            pop_start=pop, pop_end=pop, survival_rates=surv, period=(2015, 2020)
        )

        assert "period_start" in result.columns
        assert "period_end" in result.columns
        assert (result["period_start"] == 2015).all()
        assert (result["period_end"] == 2020).all()

    def test_sex_categories_match_between_pipelines(self) -> None:
        """Both pipelines use the same sex category strings."""
        from cohort_projections.data.process.residual_migration import (
            compute_residual_migration_rates,
        )

        pop = _make_population_snapshot(["38001"])
        surv = _make_survival_rates()

        residual_result = compute_residual_migration_rates(
            pop_start=pop, pop_end=pop, survival_rates=surv, period=(2020, 2024)
        )

        # The convergence pipeline expects "Male" and "Female"
        assert sorted(residual_result["sex"].unique()) == ["Female", "Male"]

    def test_age_groups_match_between_pipelines(self) -> None:
        """Residual pipeline produces the same 18 age groups convergence expects."""
        from cohort_projections.data.process.residual_migration import (
            compute_residual_migration_rates,
        )

        pop = _make_population_snapshot(["38001"])
        surv = _make_survival_rates()

        result = compute_residual_migration_rates(
            pop_start=pop, pop_end=pop, survival_rates=surv, period=(2020, 2024)
        )

        output_ags = sorted(
            result["age_group"].unique(),
            key=lambda x: AGE_GROUP_LABELS.index(x),
        )
        assert output_ags == AGE_GROUP_LABELS
