"""
Unit tests for PEP migration rate processing (process_pep_migration_rates).

Tests the Phase 3 ADR-035 pipeline that converts Census PEP county-level
net migration data into age/sex/race-specific migration tables using
regime-weighted averaging and Rogers-Castro age distribution.

Uses synthetic data fixtures -- does not depend on actual PEP data files.
The pep_regime_analysis module may not exist yet (being built concurrently),
so we inject a mock module into sys.modules before importing.
"""

import sys
import types
from unittest.mock import MagicMock, patch

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


# ---- Module-level setup: inject a stub pep_regime_analysis module ----
# This ensures the lazy import inside process_pep_migration_rates can resolve
# even when the real pep_regime_analysis.py has not been created yet.

_MODULE_NAME = "cohort_projections.data.process.pep_regime_analysis"


def _ensure_stub_module():
    """Inject a stub pep_regime_analysis module if the real one is not available."""
    try:
        # Try importing the real module first
        import importlib

        importlib.import_module(_MODULE_NAME)
    except (ImportError, ModuleNotFoundError):
        # Real module not available; inject stub
        stub = types.ModuleType(_MODULE_NAME)
        stub.DEFAULT_REGIME_WEIGHTS = {
            "pre_bakken": 0.15,
            "boom": 0.10,
            "bust_covid": 0.25,
            "recovery": 0.50,
        }
        stub.DEFAULT_DAMPENING = {"boom": 0.60}
        stub.load_pep_preferred_estimates = MagicMock()
        stub.classify_counties = MagicMock()
        stub.calculate_regime_averages = MagicMock()
        stub.calculate_regime_weighted_average = MagicMock()
        stub.generate_regime_analysis_report = MagicMock()
        # Constants expected by __init__.py imports
        stub.OIL_COUNTIES = []
        stub.METRO_COUNTIES = []
        stub.MIGRATION_REGIMES = {}
        sys.modules[_MODULE_NAME] = stub


_ensure_stub_module()

# Now safe to import from migration_rates
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


@pytest.fixture
def mock_regime_averages(synthetic_pep_data):
    """Create mock regime averages matching the expected output of calculate_regime_averages."""
    records = []
    for geoid in TEST_COUNTIES:
        county_data = synthetic_pep_data[synthetic_pep_data["geoid"] == geoid]
        for regime_name, (start, end) in [
            ("pre_bakken", (2000, 2010)),
            ("boom", (2011, 2015)),
            ("bust_covid", (2016, 2021)),
            ("recovery", (2022, 2024)),
        ]:
            regime_data = county_data[(county_data["year"] >= start) & (county_data["year"] <= end)]
            if not regime_data.empty:
                records.append(
                    {
                        "geoid": geoid,
                        "regime": regime_name,
                        "avg_netmig": regime_data["netmig"].mean(),
                        "n_years": len(regime_data),
                    }
                )
    return pd.DataFrame(records)


@pytest.fixture
def mock_weighted_averages(mock_regime_averages):
    """Create mock weighted averages matching expected output of calculate_regime_weighted_average."""
    records = []
    for geoid in TEST_COUNTIES:
        county_regimes = mock_regime_averages[mock_regime_averages["geoid"] == geoid]
        weighted_avg = county_regimes["avg_netmig"].mean()
        records.append(
            {
                "geoid": geoid,
                "weighted_avg_netmig": weighted_avg,
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def mock_county_classifications():
    """Create mock county classifications."""
    return pd.DataFrame(
        {
            "geoid": TEST_COUNTIES,
            "classification": ["oil", "metro", "rural"],
        }
    )


def _patch_pep_module(
    synthetic_pep_data,
    mock_county_classifications,
    mock_regime_averages,
    mock_weighted_averages,
):
    """Configure the stub pep_regime_analysis module with test-specific mock functions.

    Returns a context manager that patches the stub module's functions.
    """

    def mock_load(path):
        return synthetic_pep_data

    def mock_classify(pep_df):
        return mock_county_classifications

    def mock_ravg(pep_df):
        return mock_regime_averages

    def mock_wavg(regime_avgs, weights=None, dampening=None):
        return mock_weighted_averages

    # Patch the four functions on the stub module
    return _MultiPatch(
        (f"{_MODULE_NAME}.load_pep_preferred_estimates", mock_load),
        (f"{_MODULE_NAME}.classify_counties", mock_classify),
        (f"{_MODULE_NAME}.calculate_regime_averages", mock_ravg),
        (f"{_MODULE_NAME}.calculate_regime_weighted_average", mock_wavg),
    )


class _MultiPatch:  # noqa: N801
    """Context manager to apply multiple unittest.mock.patch calls at once."""

    def __init__(self, *patch_specs):
        """patch_specs: tuples of (target, side_effect)"""
        self._patches = []
        for target, side_effect in patch_specs:
            self._patches.append(patch(target, side_effect=side_effect))

    def __enter__(self):
        for p in self._patches:
            p.__enter__()
        return self

    def __exit__(self, *args):
        for p in reversed(self._patches):
            p.__exit__(*args)


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
        mock_county_classifications,
        mock_regime_averages,
        mock_weighted_averages,
    ):
        """Output has correct total number of rows (n_counties * 1,092)."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with _patch_pep_module(
            synthetic_pep_data,
            mock_county_classifications,
            mock_regime_averages,
            mock_weighted_averages,
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
        mock_county_classifications,
        mock_regime_averages,
        mock_weighted_averages,
    ):
        """Each county has exactly 1,092 rows (91 ages x 2 sexes x 6 races)."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with _patch_pep_module(
            synthetic_pep_data,
            mock_county_classifications,
            mock_regime_averages,
            mock_weighted_averages,
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
        mock_county_classifications,
        mock_regime_averages,
        mock_weighted_averages,
    ):
        """Required columns are present in output."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with _patch_pep_module(
            synthetic_pep_data,
            mock_county_classifications,
            mock_regime_averages,
            mock_weighted_averages,
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
        mock_county_classifications,
        mock_regime_averages,
        mock_weighted_averages,
    ):
        """Age distribution peaks around age 25 (Rogers-Castro pattern)."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with _patch_pep_module(
            synthetic_pep_data,
            mock_county_classifications,
            mock_regime_averages,
            mock_weighted_averages,
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
        mock_county_classifications,
        mock_regime_averages,
        mock_weighted_averages,
    ):
        """Both sexes present in roughly equal proportions."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with _patch_pep_module(
            synthetic_pep_data,
            mock_county_classifications,
            mock_regime_averages,
            mock_weighted_averages,
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
        mock_county_classifications,
        mock_regime_averages,
        mock_weighted_averages,
    ):
        """All 6 race/ethnicity categories are present in the output."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with _patch_pep_module(
            synthetic_pep_data,
            mock_county_classifications,
            mock_regime_averages,
            mock_weighted_averages,
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
        mock_county_classifications,
        mock_regime_averages,
        mock_weighted_averages,
    ):
        """Output files (parquet and CSV) are created for each scenario."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)
        output_dir = tmp_path / "output"

        with _patch_pep_module(
            synthetic_pep_data,
            mock_county_classifications,
            mock_regime_averages,
            mock_weighted_averages,
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
        mock_county_classifications,
        mock_regime_averages,
    ):
        """Run process_pep_migration_rates with all three scenarios.

        Returns results dict keyed by scenario name.
        """

        def dynamic_weighted_avg(regime_avgs, weights=None, dampening=None):
            """Produce different weighted averages depending on scenario weights."""
            records = []
            for geoid in TEST_COUNTIES:
                county_regimes = regime_avgs[regime_avgs["geoid"] == geoid]
                if county_regimes.empty:
                    records.append({"geoid": geoid, "weighted_avg_netmig": 0.0})
                    continue

                total = 0.0
                total_weight = 0.0
                for _, row in county_regimes.iterrows():
                    regime = row["regime"]
                    avg = row["avg_netmig"]
                    w = weights.get(regime, 0.0) if weights else 0.25
                    damp = dampening.get(regime, 1.0) if dampening else 1.0
                    total += avg * w * damp
                    total_weight += w

                if total_weight > 0:
                    weighted = total / total_weight
                else:
                    weighted = 0.0

                records.append({"geoid": geoid, "weighted_avg_netmig": weighted})
            return pd.DataFrame(records)

        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with (
            patch(
                f"{_MODULE_NAME}.load_pep_preferred_estimates",
                side_effect=lambda p: synthetic_pep_data,
            ),
            patch(
                f"{_MODULE_NAME}.classify_counties",
                side_effect=lambda df: mock_county_classifications,
            ),
            patch(
                f"{_MODULE_NAME}.calculate_regime_averages",
                side_effect=lambda df: mock_regime_averages,
            ),
            patch(
                f"{_MODULE_NAME}.calculate_regime_weighted_average",
                side_effect=dynamic_weighted_avg,
            ),
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
        mock_county_classifications,
        mock_regime_averages,
        mock_weighted_averages,
    ):
        """Baseline scenario is generated by default when no scenarios specified."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with _patch_pep_module(
            synthetic_pep_data,
            mock_county_classifications,
            mock_regime_averages,
            mock_weighted_averages,
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

        The low scenario applies a 0.75 multiplier to all net_migration values,
        so it should produce lower absolute totals.
        """
        results = _run_all_scenarios

        baseline_total = results["baseline"]["net_migration"].abs().sum()
        low_total = results["low"]["net_migration"].abs().sum()

        # Low should have smaller absolute values due to 0.75 multiplier
        assert low_total < baseline_total

    def test_each_scenario_has_output_files(
        self,
        tmp_path,
        synthetic_pep_data,
        synthetic_population,
        mock_config,
        mock_county_classifications,
        mock_regime_averages,
        mock_weighted_averages,
    ):
        """Each scenario produces parquet and CSV output files."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)
        output_dir = tmp_path / "output"

        with _patch_pep_module(
            synthetic_pep_data,
            mock_county_classifications,
            mock_regime_averages,
            mock_weighted_averages,
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

        mock_weighted = pd.DataFrame([{"geoid": zero_county, "weighted_avg_netmig": 0.0}])

        pep_file = tmp_path / "pep_data.parquet"
        zero_pep.to_parquet(pep_file, index=False)
        pop_file = tmp_path / "population.parquet"
        zero_pop.to_parquet(pop_file, index=False)

        with (
            patch(f"{_MODULE_NAME}.load_pep_preferred_estimates", side_effect=lambda p: zero_pep),
            patch(
                f"{_MODULE_NAME}.classify_counties",
                side_effect=lambda df: pd.DataFrame(
                    [{"geoid": zero_county, "classification": "rural"}]
                ),
            ),
            patch(
                f"{_MODULE_NAME}.calculate_regime_averages",
                side_effect=lambda df: pd.DataFrame(
                    [{"geoid": zero_county, "regime": "recovery", "avg_netmig": 0.0, "n_years": 3}]
                ),
            ),
            patch(
                f"{_MODULE_NAME}.calculate_regime_weighted_average",
                side_effect=lambda ravg, weights=None, dampening=None: mock_weighted,
            ),
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

        mock_weighted = pd.DataFrame([{"geoid": small_county, "weighted_avg_netmig": -5.0}])

        pep_file = tmp_path / "pep_data.parquet"
        small_pep.to_parquet(pep_file, index=False)
        pop_file = tmp_path / "population.parquet"
        small_pop.to_parquet(pop_file, index=False)

        with (
            patch(f"{_MODULE_NAME}.load_pep_preferred_estimates", side_effect=lambda p: small_pep),
            patch(
                f"{_MODULE_NAME}.classify_counties",
                side_effect=lambda df: pd.DataFrame(
                    [{"geoid": small_county, "classification": "rural"}]
                ),
            ),
            patch(
                f"{_MODULE_NAME}.calculate_regime_averages",
                side_effect=lambda df: pd.DataFrame(
                    [
                        {
                            "geoid": small_county,
                            "regime": "recovery",
                            "avg_netmig": -5.0,
                            "n_years": 3,
                        }
                    ]
                ),
            ),
            patch(
                f"{_MODULE_NAME}.calculate_regime_weighted_average",
                side_effect=lambda ravg, weights=None, dampening=None: mock_weighted,
            ),
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

        mock_weighted = pd.DataFrame([{"geoid": single_county, "weighted_avg_netmig": 500.0}])

        pep_file = tmp_path / "pep_data.parquet"
        single_pep.to_parquet(pep_file, index=False)
        pop_file = tmp_path / "population.parquet"
        single_pop.to_parquet(pop_file, index=False)

        with (
            patch(f"{_MODULE_NAME}.load_pep_preferred_estimates", side_effect=lambda p: single_pep),
            patch(
                f"{_MODULE_NAME}.classify_counties",
                side_effect=lambda df: pd.DataFrame(
                    [{"geoid": single_county, "classification": "metro"}]
                ),
            ),
            patch(
                f"{_MODULE_NAME}.calculate_regime_averages",
                side_effect=lambda df: pd.DataFrame(
                    [
                        {
                            "geoid": single_county,
                            "regime": "recovery",
                            "avg_netmig": 500.0,
                            "n_years": 3,
                        }
                    ]
                ),
            ),
            patch(
                f"{_MODULE_NAME}.calculate_regime_weighted_average",
                side_effect=lambda ravg, weights=None, dampening=None: mock_weighted,
            ),
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

        with pytest.raises(FileNotFoundError):
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
        mock_county_classifications,
        mock_regime_averages,
        mock_weighted_averages,
    ):
        """processing_date column is present in output."""
        pep_file, pop_file = _write_fixtures(tmp_path, synthetic_pep_data, synthetic_population)

        with _patch_pep_module(
            synthetic_pep_data,
            mock_county_classifications,
            mock_regime_averages,
            mock_weighted_averages,
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
