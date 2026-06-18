"""
Integration tests for PEP-based per-county migration rate pipeline.

Verifies that the PEP_components migration method correctly loads multi-county
parquet data, splits into per-county dicts, applies scenario adjustments,
and feeds into the geographic projection runner.

Uses synthetic data and tmp_path -- no actual project data files required.
"""

import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# The pipeline script starts with a digit (02_run_projections.py) so it cannot
# be imported with a normal import statement.  Use importlib to load it.
_spec = importlib.util.spec_from_file_location(
    "run_projections_02",
    PROJECT_ROOT / "scripts" / "pipeline" / "02_run_projections.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_create_zero_migration_rates = _mod._create_zero_migration_rates
_transform_migration_rates = _mod._transform_migration_rates
apply_scenario_rate_adjustments = _mod.apply_scenario_rate_adjustments
load_demographic_rates = _mod.load_demographic_rates
aggregate_county_results_to_state = _mod.aggregate_county_results_to_state

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGES = list(range(91))
SEXES = ["Male", "Female"]
RACES = [
    "White alone, Non-Hispanic",
    "Black alone, Non-Hispanic",
    "AIAN alone, Non-Hispanic",
    "Asian/PI alone, Non-Hispanic",
    "Two or more races, Non-Hispanic",
    "Hispanic (any race)",
]
ROWS_PER_COUNTY = len(AGES) * len(SEXES) * len(RACES)  # 1,092
TEST_COUNTIES = ["38017", "38101", "38015"]  # Cass, Ward, Burleigh


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_county_migration_df(county_fips: str, base_rate: float = 0.002) -> pd.DataFrame:
    """Build a single county's migration rate rows (1,092 rows)."""
    rows = [
        {
            "county_fips": county_fips,
            "age": age,
            "sex": sex,
            "race": race,
            "migration_rate": base_rate,
        }
        for age in AGES
        for sex in SEXES
        for race in RACES
    ]
    return pd.DataFrame(rows)


def _setup_processed_dir(tmp_path: Path, fertility_parquet: Path, survival_parquet: Path) -> Path:
    """Create data/processed/ under tmp_path and copy fertility/survival files."""
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    shutil.copy(fertility_parquet, proc / "fertility_rates.parquet")
    shutil.copy(survival_parquet, proc / "survival_rates.parquet")
    return proc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pep_parquet(tmp_path: Path) -> Path:
    """Create a synthetic multi-county PEP migration rates parquet file."""
    frames = [
        _build_county_migration_df("38017", base_rate=0.003),
        _build_county_migration_df("38101", base_rate=0.001),
        _build_county_migration_df("38015", base_rate=-0.002),
    ]
    combined = pd.concat(frames, ignore_index=True)
    out = tmp_path / "migration_rates_pep_baseline.parquet"
    combined.to_parquet(out, index=False)
    return out


@pytest.fixture
def fertility_parquet(tmp_path: Path) -> Path:
    """Create a minimal fertility rates parquet file."""
    rows = [
        {"age": age, "race_ethnicity": race_code, "asfr": 60.0, "year": 2022}
        for age in range(15, 50)
        for race_code in ["white_nh", "black_nh", "aian_nh", "asian_nh", "hispanic"]
    ]
    df = pd.DataFrame(rows)
    out = tmp_path / "fertility_rates.parquet"
    df.to_parquet(out, index=False)
    return out


@pytest.fixture
def survival_parquet(tmp_path: Path) -> Path:
    """Create a minimal survival rates parquet file."""
    rows = [
        {
            "age": age,
            "sex": sex,
            "race_ethnicity": race_code,
            "survival_rate": max(0.5, 1.0 - age * 0.002),
        }
        for age in AGES
        for sex in ["female", "male"]
        for race_code in ["white_nh", "black_nh", "aian_nh", "asian_nh", "hispanic"]
    ]
    df = pd.DataFrame(rows)
    out = tmp_path / "survival_rates.parquet"
    df.to_parquet(out, index=False)
    return out


@pytest.fixture
def pep_config(tmp_path: Path, pep_parquet: Path):
    """Minimal config dict that selects the PEP_components method."""
    return {
        "rates": {
            "migration": {
                "domestic": {
                    "method": "PEP_components",
                },
            },
        },
        "pipeline": {
            "data_processing": {
                "migration": {
                    "pep_output": str(pep_parquet),
                },
            },
        },
        "scenarios": {
            "baseline": {
                "name": "Baseline",
                "fertility": "constant",
                "mortality": "constant",
                "migration": "recent_average",
                "active": True,
            },
            "restricted_growth": {
                "name": "Restricted Growth",
                "fertility": "-5_percent",
                "mortality": "constant",
                "migration": "cbo_time_varying",
                "active": False,
            },
            "high_growth": {
                "name": "High Growth",
                "fertility": "+5_percent",
                "mortality": "constant",
                "migration": "recent_average",
                "active": False,
            },
            "zero_migration": {
                "name": "Zero Migration",
                "fertility": "constant",
                "mortality": "constant",
                "migration": "zero",
                "active": False,
            },
        },
    }


@pytest.fixture
def irs_config():
    """Config dict that uses the legacy IRS_county_flows method."""
    return {
        "rates": {
            "migration": {
                "domestic": {
                    "method": "IRS_county_flows",
                },
            },
        },
        "scenarios": {
            "baseline": {
                "name": "Baseline",
                "fertility": "constant",
                "mortality": "constant",
                "migration": "recent_average",
                "active": True,
            },
        },
    }


# ---------------------------------------------------------------------------
# Tests: load_demographic_rates with PEP
# ---------------------------------------------------------------------------


class TestLoadDemographicRatesPep:
    """Verify load_demographic_rates branches correctly for PEP_components."""

    def test_pep_returns_dict_for_migration(
        self, monkeypatch, tmp_path, pep_parquet, fertility_parquet, survival_parquet, pep_config
    ):
        """When method is PEP_components, migration_rates is a dict."""
        _setup_processed_dir(tmp_path, fertility_parquet, survival_parquet)
        monkeypatch.setattr(_mod, "project_root", tmp_path)

        # allow_static_survival=True: these PEP tests deliberately run with only the
        # static-base survival table (no operative nd_adjusted_survival_projections),
        # which is a hard error in production after the ADR-068 2026-06-16 hardening.
        _fert, _surv, migration, _mig_by_yr, _surv_by_yr = load_demographic_rates(
            pep_config, allow_static_survival=True
        )

        assert isinstance(migration, dict), "PEP method should return a dict"

    def test_pep_dict_has_correct_keys(
        self, monkeypatch, tmp_path, pep_parquet, fertility_parquet, survival_parquet, pep_config
    ):
        """Dict keys should be the county FIPS codes from the parquet."""
        _setup_processed_dir(tmp_path, fertility_parquet, survival_parquet)
        monkeypatch.setattr(_mod, "project_root", tmp_path)

        # allow_static_survival=True: these PEP tests deliberately run with only the
        # static-base survival table (no operative nd_adjusted_survival_projections),
        # which is a hard error in production after the ADR-068 2026-06-16 hardening.
        _fert, _surv, migration, _mig_by_yr, _surv_by_yr = load_demographic_rates(
            pep_config, allow_static_survival=True
        )

        assert set(migration.keys()) == set(TEST_COUNTIES)

    def test_pep_each_county_has_correct_shape(
        self, monkeypatch, tmp_path, pep_parquet, fertility_parquet, survival_parquet, pep_config
    ):
        """Each county DataFrame should have 1,092 rows."""
        _setup_processed_dir(tmp_path, fertility_parquet, survival_parquet)
        monkeypatch.setattr(_mod, "project_root", tmp_path)

        # allow_static_survival=True: these PEP tests deliberately run with only the
        # static-base survival table (no operative nd_adjusted_survival_projections),
        # which is a hard error in production after the ADR-068 2026-06-16 hardening.
        _fert, _surv, migration, _mig_by_yr, _surv_by_yr = load_demographic_rates(
            pep_config, allow_static_survival=True
        )

        for fips, df in migration.items():
            assert len(df) == ROWS_PER_COUNTY, (
                f"County {fips} has {len(df)} rows, expected {ROWS_PER_COUNTY}"
            )

    def test_pep_columns_renamed_correctly(
        self, monkeypatch, tmp_path, pep_parquet, fertility_parquet, survival_parquet, pep_config
    ):
        """Columns should include 'race' and not 'county_fips' after transform."""
        _setup_processed_dir(tmp_path, fertility_parquet, survival_parquet)
        monkeypatch.setattr(_mod, "project_root", tmp_path)

        # allow_static_survival=True: these PEP tests deliberately run with only the
        # static-base survival table (no operative nd_adjusted_survival_projections),
        # which is a hard error in production after the ADR-068 2026-06-16 hardening.
        _fert, _surv, migration, _mig_by_yr, _surv_by_yr = load_demographic_rates(
            pep_config, allow_static_survival=True
        )

        sample_df = next(iter(migration.values()))
        assert "race" in sample_df.columns
        assert "county_fips" not in sample_df.columns


# ---------------------------------------------------------------------------
# Tests: _transform_migration_rates dict handling
# ---------------------------------------------------------------------------


class TestTransformMigrationRatesDict:
    """Verify _transform_migration_rates handles both dict and DataFrame."""

    def test_dict_input_produces_dict_output(self):
        """A dict of DataFrames should produce a dict of transformed DataFrames."""
        df = _build_county_migration_df("38017").drop(columns=["county_fips"])
        result = _transform_migration_rates({"38017": df})
        assert isinstance(result, dict)
        assert "38017" in result

    def test_each_value_properly_transformed(self):
        """Each transformed DataFrame should have the engine-expected columns."""
        df = _build_county_migration_df("38017").drop(columns=["county_fips"])
        result = _transform_migration_rates({"38017": df})
        transformed = result["38017"]
        assert "age" in transformed.columns
        assert "sex" in transformed.columns
        assert "race" in transformed.columns

    def test_single_dataframe_still_works(self):
        """Backward compat: a plain DataFrame input returns a DataFrame."""
        df = _build_county_migration_df("38017").drop(columns=["county_fips"])
        result = _transform_migration_rates(df)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: scenario adjustments with dict migration
# ---------------------------------------------------------------------------


class TestScenarioAdjustmentsDict:
    """Verify scenario preparation passes dict migration rates to the engine."""

    @pytest.fixture
    def base_rates(self):
        """Create base fertility, survival, and dict migration rates."""
        fertility = pd.DataFrame(
            {
                "age": [25, 30],
                "race": ["White alone, Non-Hispanic"] * 2,
                "fertility_rate": [0.06, 0.08],
            }
        )
        survival = pd.DataFrame(
            {
                "age": [25, 30],
                "sex": ["Female", "Male"],
                "race": ["White alone, Non-Hispanic"] * 2,
                "survival_rate": [0.999, 0.998],
            }
        )
        migration = {
            "38017": pd.DataFrame(
                {
                    "age": [25, 30],
                    "sex": ["Female", "Male"],
                    "race": ["White alone, Non-Hispanic"] * 2,
                    "migration_rate": [0.004, 0.002],
                }
            ),
            "38101": pd.DataFrame(
                {
                    "age": [25, 30],
                    "sex": ["Female", "Male"],
                    "race": ["White alone, Non-Hispanic"] * 2,
                    "migration_rate": [0.001, 0.001],
                }
            ),
        }
        return fertility, survival, migration

    def test_plus_25_percent_passes_rates_through(self, base_rates):
        """'+25_percent' is not pre-applied before engine execution."""
        fert, surv, mig = base_rates
        config = {
            "scenarios": {
                "high": {
                    "migration": "+25_percent",
                    "fertility": "constant",
                    "mortality": "constant",
                }
            }
        }
        _, _, adj, _, _ = apply_scenario_rate_adjustments("high", config, fert, surv, mig)
        assert isinstance(adj, dict)
        for fips in adj:
            pd.testing.assert_frame_equal(adj[fips], mig[fips])

    def test_minus_25_percent_passes_rates_through(self, base_rates):
        """'-25_percent' is not pre-applied before engine execution."""
        fert, surv, mig = base_rates
        config = {
            "scenarios": {
                "low": {
                    "migration": "-25_percent",
                    "fertility": "constant",
                    "mortality": "constant",
                }
            }
        }
        _, _, adj, _, _ = apply_scenario_rate_adjustments("low", config, fert, surv, mig)
        assert isinstance(adj, dict)
        for fips in adj:
            pd.testing.assert_frame_equal(adj[fips], mig[fips])

    def test_zero_migration_passes_rates_through(self, base_rates):
        """'zero' is not pre-applied before engine execution."""
        fert, surv, mig = base_rates
        config = {
            "scenarios": {
                "zero": {
                    "migration": "zero",
                    "fertility": "constant",
                    "mortality": "constant",
                }
            }
        }
        _, _, adj, _, _ = apply_scenario_rate_adjustments("zero", config, fert, surv, mig)
        assert isinstance(adj, dict)
        for fips in adj:
            pd.testing.assert_frame_equal(adj[fips], mig[fips])

    def test_fertility_adjustment_passes_rates_through(self, base_rates):
        """Fertility multipliers are not pre-applied before engine execution."""
        fert, surv, mig = base_rates
        config = {
            "scenarios": {
                "cbo_adjusted": {
                    "migration": "recent_average",
                    "fertility": "-5_percent",
                    "mortality": "improving",
                }
            }
        }
        adj_fert, _, _, _, _ = apply_scenario_rate_adjustments(
            "cbo_adjusted", config, fert, surv, mig
        )
        pd.testing.assert_frame_equal(adj_fert, fert)

    def test_recent_average_leaves_unchanged(self, base_rates):
        """'recent_average' should not modify rates."""
        fert, surv, mig = base_rates
        config = {
            "scenarios": {
                "baseline": {
                    "migration": "recent_average",
                    "fertility": "constant",
                    "mortality": "constant",
                }
            }
        }
        _, _, adj, _, _ = apply_scenario_rate_adjustments("baseline", config, fert, surv, mig)
        assert isinstance(adj, dict)
        for fips in adj:
            pd.testing.assert_frame_equal(adj[fips], mig[fips])


# ---------------------------------------------------------------------------
# Tests: run_geographic_projections with PEP per-county rates
# ---------------------------------------------------------------------------


class TestRunGeographicProjectionsPep:
    """Verify migration_rates_by_geography is built correctly."""

    def test_per_county_rates_used_when_dict(self):
        """When adj_migration is a dict, per-county rates should be used."""
        migration_dict = {
            "38017": _build_county_migration_df("38017").drop(columns=["county_fips"]),
            "38101": _build_county_migration_df("38101").drop(columns=["county_fips"]),
        }
        fips_to_process = ["38017", "38101"]

        # Simulate the logic from run_geographic_projections
        default_migration = _create_zero_migration_rates()
        migration_rates_by_geography = {
            fips: migration_dict.get(fips, default_migration) for fips in fips_to_process
        }

        assert set(migration_rates_by_geography.keys()) == {"38017", "38101"}
        for fips in fips_to_process:
            assert len(migration_rates_by_geography[fips]) == ROWS_PER_COUNTY

    def test_missing_county_gets_zero_migration(self):
        """Counties not in the PEP dict should receive zero migration rates."""
        migration_dict = {
            "38017": _build_county_migration_df("38017").drop(columns=["county_fips"]),
        }
        fips_to_process = ["38017", "38999"]  # 38999 is not in dict

        default_migration = _create_zero_migration_rates()
        migration_rates_by_geography = {
            fips: migration_dict.get(fips, default_migration) for fips in fips_to_process
        }

        # 38999 should get zero rates
        missing_rates = migration_rates_by_geography["38999"]
        assert (missing_rates["migration_rate"] == 0.0).all()

        # 38017 should keep its original rates
        present_rates = migration_rates_by_geography["38017"]
        assert (present_rates["migration_rate"] != 0.0).any()

    def test_single_dataframe_behavior_preserved(self):
        """Existing single-DataFrame path should still produce uniform rates."""
        single_df = _create_zero_migration_rates()
        single_df["migration_rate"] = 0.005  # non-zero baseline

        fips_to_process = ["38017", "38101"]

        # Simulate existing IRS path (dict.fromkeys)
        migration_rates_by_geography = dict.fromkeys(fips_to_process, single_df)

        # All geographies should share the same DataFrame object
        assert migration_rates_by_geography["38017"] is migration_rates_by_geography["38101"]
        assert (migration_rates_by_geography["38017"]["migration_rate"] == 0.005).all()

    def test_zero_migration_rates_shape(self):
        """_create_zero_migration_rates should return 1,092 rows."""
        zero_df = _create_zero_migration_rates()
        assert len(zero_df) == ROWS_PER_COUNTY
        assert set(zero_df.columns) == {"age", "sex", "race", "migration_rate"}
        assert (zero_df["migration_rate"] == 0.0).all()


# ---------------------------------------------------------------------------
# ADR-068 recurrence guards (2026-06-16 PR #25 hardening)
# ---------------------------------------------------------------------------


def _write_operative_survival(processed_dir: Path, years: range) -> Path:
    """Write a minimal but valid operative survival table spanning ``years``.

    Schema matches Phase 3 output ([year, age, sex, survival_rate]) so
    ``_build_survival_rates_by_year`` consumes it. Used to simulate a
    horizon-truncated operative table (the ADR-068 defect).
    """
    mortality_dir = processed_dir / "mortality"
    mortality_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"year": year, "age": age, "sex": sex, "survival_rate": 0.99, "source": "test"}
        for year in years
        for age in range(101)
        for sex in ["Male", "Female"]
    ]
    out = mortality_dir / "nd_adjusted_survival_projections.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


class TestSurvivalCoverageGuard:
    """M1 (ADR-068): the survival-coverage guard hard-fails by default.

    A missing or horizon-incomplete operative survival table would otherwise let the
    engine silently fall back to static-base survival (the 2026-06-15 truncation bug).
    """

    def test_missing_operative_table_raises_by_default(
        self, monkeypatch, tmp_path, fertility_parquet, survival_parquet, pep_config
    ):
        """No operative survival table + no opt-out -> RuntimeError (production default)."""
        _setup_processed_dir(tmp_path, fertility_parquet, survival_parquet)
        monkeypatch.setattr(_mod, "project_root", tmp_path)
        with pytest.raises(RuntimeError, match="No operative mortality-improvement survival"):
            load_demographic_rates(pep_config)

    def test_missing_operative_table_allowed_with_optout(
        self, monkeypatch, tmp_path, fertility_parquet, survival_parquet, pep_config
    ):
        """No operative table + allow_static_survival=True -> no raise (tests/experiments)."""
        _setup_processed_dir(tmp_path, fertility_parquet, survival_parquet)
        monkeypatch.setattr(_mod, "project_root", tmp_path)
        # Should not raise.
        load_demographic_rates(pep_config, allow_static_survival=True)

    def test_incomplete_operative_table_raises_by_default(
        self, monkeypatch, tmp_path, fertility_parquet, survival_parquet, pep_config
    ):
        """A truncated operative table (2025-2030, horizon needs 2055) -> RuntimeError."""
        proc = _setup_processed_dir(tmp_path, fertility_parquet, survival_parquet)
        _write_operative_survival(proc, range(2025, 2031))  # truncated, missing 2031-2055
        monkeypatch.setattr(_mod, "project_root", tmp_path)
        cfg = {**pep_config, "project": {"base_year": 2025, "projection_horizon": 30}}
        with pytest.raises(RuntimeError, match="OPERATIVE SURVIVAL COVERAGE GAP"):
            load_demographic_rates(cfg)

    def test_incomplete_operative_table_allowed_with_optout(
        self, monkeypatch, tmp_path, fertility_parquet, survival_parquet, pep_config
    ):
        """A truncated operative table + opt-out -> warns, does not raise."""
        proc = _setup_processed_dir(tmp_path, fertility_parquet, survival_parquet)
        _write_operative_survival(proc, range(2025, 2031))
        monkeypatch.setattr(_mod, "project_root", tmp_path)
        cfg = {**pep_config, "project": {"base_year": 2025, "projection_horizon": 30}}
        # Should not raise.
        load_demographic_rates(cfg, allow_static_survival=True)


def _write_convergence(processed_dir: Path, offsets_by_county: dict[str, range | list[int]]) -> Path:
    """Write a minimal convergence_rates_by_year.parquet with per-county year-offsets.

    Schema matches Phase 2 output ([year_offset, county_fips, age_group, sex,
    migration_rate]); a single valid 5-year age group per (county, offset) is enough for
    ``_build_convergence_rate_dicts`` to register the offset. Used to simulate a
    horizon-truncated convergence table (the ADR-068 survival-truncation twin on the
    migration driver). An ``annual_rate`` metadata sidecar is written so the loader skips
    the legacy annualization path.
    """
    migration_dir = processed_dir / "migration"
    migration_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "year_offset": int(off),
            "county_fips": fips,
            "age_group": "25-29",
            "sex": sex,
            "migration_rate": 0.001,
        }
        for fips, offsets in offsets_by_county.items()
        for off in offsets
        for sex in SEXES
    ]
    out = migration_dir / "convergence_rates_by_year.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    (migration_dir / "convergence_metadata.json").write_text('{"rate_unit": "annual_rate"}')
    return out


class TestConvergenceCoverageGuard:
    """ADR-068 recurrence-hardening: the convergence migration-coverage guard.

    A convergence table that is present but covers fewer year-offsets than the configured
    horizon would let the engine silently fall back to constant migration rates for the
    tail years — the survival truncation defect's twin, on the projection's dominant
    driver. The guard hard-fails by default and is silent on a full-horizon table.

    ``allow_static_survival=True`` is passed throughout to isolate the migration guard
    (the migration check runs before the survival check in ``load_demographic_rates``).
    """

    def test_truncated_convergence_raises_by_default(
        self, monkeypatch, tmp_path, fertility_parquet, survival_parquet, pep_config
    ):
        """Convergence covering offsets 1-20 with a 30-year horizon -> RuntimeError."""
        proc = _setup_processed_dir(tmp_path, fertility_parquet, survival_parquet)
        _write_convergence(proc, {fips: range(1, 21) for fips in TEST_COUNTIES})  # missing 21-30
        monkeypatch.setattr(_mod, "project_root", tmp_path)
        cfg = {**pep_config, "project": {"base_year": 2025, "projection_horizon": 30}}
        with pytest.raises(RuntimeError, match="CONVERGENCE MIGRATION COVERAGE GAP"):
            load_demographic_rates(cfg, allow_static_survival=True)

    def test_truncated_convergence_allowed_with_optout(
        self, monkeypatch, tmp_path, fertility_parquet, survival_parquet, pep_config
    ):
        """Truncated convergence + allow_static_migration=True -> warns, does not raise."""
        proc = _setup_processed_dir(tmp_path, fertility_parquet, survival_parquet)
        _write_convergence(proc, {fips: range(1, 21) for fips in TEST_COUNTIES})
        monkeypatch.setattr(_mod, "project_root", tmp_path)
        cfg = {**pep_config, "project": {"base_year": 2025, "projection_horizon": 30}}
        # Should not raise.
        load_demographic_rates(cfg, allow_static_survival=True, allow_static_migration=True)

    def test_full_horizon_convergence_passes(
        self, monkeypatch, tmp_path, fertility_parquet, survival_parquet, pep_config
    ):
        """A convergence table spanning offsets 1-30 for a 30-year horizon -> no raise."""
        proc = _setup_processed_dir(tmp_path, fertility_parquet, survival_parquet)
        _write_convergence(proc, {fips: range(1, 31) for fips in TEST_COUNTIES})
        monkeypatch.setattr(_mod, "project_root", tmp_path)
        cfg = {**pep_config, "project": {"base_year": 2025, "projection_horizon": 30}}
        # Should not raise — guard is silent on a full-horizon table.
        load_demographic_rates(cfg, allow_static_survival=True)

    def test_single_short_county_raises(
        self, monkeypatch, tmp_path, fertility_parquet, survival_parquet, pep_config
    ):
        """One short county among otherwise-complete counties still trips the guard."""
        proc = _setup_processed_dir(tmp_path, fertility_parquet, survival_parquet)
        offsets = {fips: range(1, 31) for fips in TEST_COUNTIES}
        offsets[TEST_COUNTIES[1]] = range(1, 21)  # one county truncated
        _write_convergence(proc, offsets)
        monkeypatch.setattr(_mod, "project_root", tmp_path)
        cfg = {**pep_config, "project": {"base_year": 2025, "projection_horizon": 30}}
        with pytest.raises(RuntimeError, match="CONVERGENCE MIGRATION COVERAGE GAP"):
            load_demographic_rates(cfg, allow_static_survival=True)


def _write_county_parquet(county_dir: Path, fips: str, base: int, end: int, scenario: str) -> Path:
    """Write a minimal county projection parquet with the standard filename convention."""
    county_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [{"year": base, "age": 0, "sex": "Male", "race": RACES[0], "population": 1.0}]
    )
    out = county_dir / f"nd_county_{fips}_projection_{base}_{end}_{scenario}.parquet"
    df.to_parquet(out, index=False)
    return out


class TestStateAggregationGuards:
    """M2 (ADR-068): state aggregation rejects a stale-horizon or partial county set.

    A complete set of stale 2025-2045 county files (or a missing county) would pass the
    old duplicate-only check and be silently summed into a mislabeled state total.
    """

    def _list_config(self, fips_codes):
        return {"geography": {"counties": {"mode": "list", "fips_codes": fips_codes}}}

    def test_stale_horizon_files_fail(self, tmp_path):
        """County files with a horizon != the run's horizon -> aggregation fails."""
        county_dir = tmp_path / "baseline" / "county"
        _write_county_parquet(county_dir, "38001", 2025, 2045, "baseline")  # stale
        _write_county_parquet(county_dir, "38003", 2025, 2045, "baseline")  # stale
        config = self._list_config(["38001", "38003"])
        config["project"] = {"base_year": 2025, "projection_horizon": 30}  # run is 2025-2055
        assert aggregate_county_results_to_state(tmp_path, "baseline", config) is False

    def test_missing_county_fails(self, tmp_path):
        """A current-horizon set missing an expected county -> aggregation fails."""
        county_dir = tmp_path / "baseline" / "county"
        _write_county_parquet(county_dir, "38001", 2025, 2055, "baseline")  # only one of two
        config = self._list_config(["38001", "38003"])
        config["project"] = {"base_year": 2025, "projection_horizon": 30}
        assert aggregate_county_results_to_state(tmp_path, "baseline", config) is False
