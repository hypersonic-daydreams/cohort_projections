"""
Tests for PP-003 IMP-07 place projection configuration.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cohort_projections.data.process.place_projection_orchestrator import run_place_projections
from cohort_projections.data.process.place_share_trending import trend_all_places_in_county
from cohort_projections.utils import load_projection_config


def _repo_root() -> Path:
    """Return repository root from this test module path."""
    return Path(__file__).resolve().parents[2]


def _load_repo_projection_config() -> dict:
    """Load the repository's canonical projection config file."""
    config_path = _repo_root() / "config" / "projection_config.yaml"
    config = load_projection_config(config_path)
    assert isinstance(config, dict)
    return config


def test_projection_config_has_place_projection_block_with_expected_defaults() -> None:
    """Config contains IMP-07 place_projections fields and expected defaults."""
    config = _load_repo_projection_config()
    assert "place_projections" in config

    place_cfg = config["place_projections"]
    assert place_cfg["enabled"] is True
    assert place_cfg["crosswalk_path"] == "data/processed/geographic/place_county_crosswalk_2020.csv"
    assert place_cfg["historical_shares_path"] == "data/processed/place_shares_2000_2024.parquet"
    assert (
        place_cfg["place_population_history_path"]
        == "data/processed/place_population_history_2000_2024.parquet"
    )

    assert place_cfg["model"]["epsilon"] == 0.001
    assert place_cfg["model"]["lambda_decay"] == 0.9
    assert place_cfg["model"]["history_start"] == 2000
    assert place_cfg["model"]["history_end"] == 2024
    assert place_cfg["model"]["reconciliation_flag_threshold"] == 0.05

    assert place_cfg["tiers"]["high_threshold"] == 10000
    assert place_cfg["tiers"]["moderate_threshold"] == 2500
    assert place_cfg["tiers"]["lower_threshold"] == 500
    assert place_cfg["tiers"]["tier_boundary_margin"] == 0.05

    assert place_cfg["backtest"]["primary_train"] == [2000, 2014]
    assert place_cfg["backtest"]["primary_test"] == [2015, 2024]
    assert place_cfg["backtest"]["secondary_train"] == [2000, 2019]
    assert place_cfg["backtest"]["secondary_test"] == [2020, 2024]

    assert place_cfg["output"]["base_year"] == 2025
    assert place_cfg["output"]["end_year"] == 2055
    assert place_cfg["output"]["key_years"] == [2025, 2030, 2035, 2040, 2045, 2050, 2055]


def test_place_share_trending_consumes_place_projection_year_window() -> None:
    """IMP-05 share trending reads projection year bounds from place_projections.output."""
    config = _load_repo_projection_config()

    place_history = pd.DataFrame(
        [
            {"county_fips": "38017", "place_fips": "3825700", "year": 2020, "row_type": "place", "share_raw": 0.44},
            {"county_fips": "38017", "place_fips": "3825700", "year": 2021, "row_type": "place", "share_raw": 0.45},
            {"county_fips": "38017", "place_fips": "3825700", "year": 2022, "row_type": "place", "share_raw": 0.46},
            {"county_fips": "38017", "place_fips": "3825700", "year": 2023, "row_type": "place", "share_raw": 0.47},
            {"county_fips": "38017", "place_fips": "3825700", "year": 2024, "row_type": "place", "share_raw": 0.48},
        ]
    )
    county_pop_history = pd.DataFrame(
        {
            "year": list(range(2025, 2056)),
            "county_population": [1000.0] * 31,
        }
    )

    projected = trend_all_places_in_county(
        place_share_history=place_history,
        county_pop_history=county_pop_history,
        config=config,
    )

    assert int(projected["year"].min()) == 2025
    assert int(projected["year"].max()) == 2055
    assert set(projected["fitting_method"].unique()) == {"ols"}
    assert set(projected["constraint_method"].unique()) == {"proportional"}


def test_place_projection_orchestrator_consumes_config_paths(tmp_path: Path) -> None:
    """IMP-06 orchestrator reads data paths from place_projections config block."""
    config = _load_repo_projection_config()

    projection_root = tmp_path / "projections"
    county_dir = projection_root / "baseline" / "county"
    county_dir.mkdir(parents=True, exist_ok=True)

    county_rows: list[dict[str, object]] = []
    for year, total in [(2025, 1000.0), (2026, 1020.0)]:
        per_cell = total / (91 * 2)
        for age in range(91):
            for sex in ["Male", "Female"]:
                county_rows.append(
                    {
                        "year": year,
                        "age": age,
                        "sex": sex,
                        "race": "White alone, Non-Hispanic",
                        "population": per_cell,
                    }
                )
    county_df = pd.DataFrame(county_rows)
    county_df.to_parquet(
        county_dir / "nd_county_38017_projection_2025_2026_baseline.parquet",
        index=False,
    )

    crosswalk = pd.DataFrame(
        [
            {
                "state_fips": 38,
                "place_fips": 3890000,
                "place_name": "Smallville city",
                "county_fips": 38017,
                "assignment_type": "single_county",
                "area_share": 1.0,
                "historical_only": False,
                "source_vintage": 2020,
                "source_method": "test",
                "confidence_tier": "LOWER",
                "tier_boundary": False,
            }
        ]
    )
    crosswalk_path = tmp_path / "place_county_crosswalk_2020.csv"
    crosswalk.to_csv(crosswalk_path, index=False)

    shares = pd.DataFrame(
        [
            {"county_fips": "38017", "place_fips": "3890000", "year": 2020, "row_type": "place", "share_raw": 0.03},
            {"county_fips": "38017", "place_fips": "3890000", "year": 2021, "row_type": "place", "share_raw": 0.03},
            {"county_fips": "38017", "place_fips": "3890000", "year": 2022, "row_type": "place", "share_raw": 0.03},
            {"county_fips": "38017", "place_fips": "3890000", "year": 2023, "row_type": "place", "share_raw": 0.03},
            {"county_fips": "38017", "place_fips": "3890000", "year": 2024, "row_type": "place", "share_raw": 0.03},
        ]
    )
    shares_path = tmp_path / "place_shares_2000_2024.parquet"
    shares.to_parquet(shares_path, index=False)

    counties = pd.DataFrame(
        [{"county_fips": 38017, "state_fips": 38, "county_name": "Cass County"}]
    )
    counties_path = tmp_path / "nd_counties.csv"
    counties.to_csv(counties_path, index=False)

    config["pipeline"]["projection"]["output_dir"] = str(projection_root)
    config["geography"]["reference_data"]["counties_file"] = str(counties_path)
    config["place_projections"]["crosswalk_path"] = str(crosswalk_path)
    config["place_projections"]["historical_shares_path"] = str(shares_path)
    config["place_projections"]["output"]["base_year"] = 2025
    config["place_projections"]["output"]["end_year"] = 2026

    result = run_place_projections(
        scenario="baseline",
        config=config,
        variant_winner="A-I",
    )

    assert result["places_processed"] == 1
    place_dir = projection_root / "baseline" / "place"
    assert (place_dir / "nd_place_3890000_projection_2025_2026_baseline.parquet").exists()
    assert (place_dir / "nd_place_3890000_projection_2025_2026_baseline_metadata.json").exists()
    assert (place_dir / "nd_place_3890000_projection_2025_2026_baseline_summary.csv").exists()
    assert (place_dir / "places_summary.csv").exists()
    assert (place_dir / "places_metadata.json").exists()

