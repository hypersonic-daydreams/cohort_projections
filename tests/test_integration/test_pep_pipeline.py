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

        _fert, _surv, migration, _mig_by_yr, _surv_by_yr = load_demographic_rates(pep_config)

        assert isinstance(migration, dict), "PEP method should return a dict"

    def test_pep_dict_has_correct_keys(
        self, monkeypatch, tmp_path, pep_parquet, fertility_parquet, survival_parquet, pep_config
    ):
        """Dict keys should be the county FIPS codes from the parquet."""
        _setup_processed_dir(tmp_path, fertility_parquet, survival_parquet)
        monkeypatch.setattr(_mod, "project_root", tmp_path)

        _fert, _surv, migration, _mig_by_yr, _surv_by_yr = load_demographic_rates(pep_config)

        assert set(migration.keys()) == set(TEST_COUNTIES)

    def test_pep_each_county_has_correct_shape(
        self, monkeypatch, tmp_path, pep_parquet, fertility_parquet, survival_parquet, pep_config
    ):
        """Each county DataFrame should have 1,092 rows."""
        _setup_processed_dir(tmp_path, fertility_parquet, survival_parquet)
        monkeypatch.setattr(_mod, "project_root", tmp_path)

        _fert, _surv, migration, _mig_by_yr, _surv_by_yr = load_demographic_rates(pep_config)

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

        _fert, _surv, migration, _mig_by_yr, _surv_by_yr = load_demographic_rates(pep_config)

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
    """Verify apply_scenario_rate_adjustments handles dict migration rates."""

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

    def test_plus_25_percent(self, base_rates):
        """'+25_percent' applies 1.25x to all counties."""
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
            original = mig[fips]["migration_rate"].values
            adjusted = adj[fips]["migration_rate"].values
            for o, a in zip(original, adjusted, strict=True):
                assert abs(a - o * 1.25) < 1e-9

    def test_minus_25_percent(self, base_rates):
        """'-25_percent' applies 0.75x to all counties."""
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
            original = mig[fips]["migration_rate"].values
            adjusted = adj[fips]["migration_rate"].values
            for o, a in zip(original, adjusted, strict=True):
                assert abs(a - o * 0.75) < 1e-9

    def test_zero_migration(self, base_rates):
        """'zero' sets all migration rates to 0."""
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
            assert (adj[fips]["migration_rate"] == 0.0).all()

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
