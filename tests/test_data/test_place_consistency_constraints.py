"""
Tests for PP-003 IMP-13 consistency constraint enforcement.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import cohort_projections.data.process.place_projection_orchestrator as orchestrator


def _synthetic_county_cohorts(years: list[int], totals_by_year: dict[int, float]) -> pd.DataFrame:
    """Build synthetic county cohort rows for projection inputs."""
    rows: list[dict[str, object]] = []
    ages = list(range(91))
    sexes = ["Male", "Female"]
    cell_count = len(ages) * len(sexes)
    for year in years:
        cell_value = totals_by_year[year] / cell_count
        for age in ages:
            for sex in sexes:
                rows.append(
                    {
                        "year": year,
                        "age": age,
                        "sex": sex,
                        "population": cell_value,
                    }
                )
    return pd.DataFrame(rows)


def _minimal_crosswalk(tmp_path: Path) -> Path:
    """Write one-place projected crosswalk and return its path."""
    crosswalk = pd.DataFrame(
        [
            {
                "state_fips": 38,
                "place_fips": 3825700,
                "place_name": "Fargo city",
                "county_fips": 38017,
                "assignment_type": "single_county",
                "area_share": 1.0,
                "historical_only": False,
                "source_vintage": 2020,
                "source_method": "test",
                "confidence_tier": "HIGH",
                "tier_boundary": False,
            }
        ]
    )
    crosswalk_path = tmp_path / "place_county_crosswalk_2020.csv"
    crosswalk.to_csv(crosswalk_path, index=False)
    return crosswalk_path


def _minimal_share_history(tmp_path: Path) -> Path:
    """Write one-place historical shares and return parquet path."""
    shares = pd.DataFrame(
        [
            {
                "county_fips": "38017",
                "place_fips": "3825700",
                "year": year,
                "row_type": "place",
                "share_raw": share,
            }
            for year, share in [(2020, 0.60), (2021, 0.61), (2022, 0.62), (2023, 0.63), (2024, 0.64)]
        ]
    )
    shares_path = tmp_path / "place_shares_2000_2024.parquet"
    shares.to_parquet(shares_path, index=False)
    return shares_path


def _minimal_counties_file(tmp_path: Path) -> Path:
    """Write county names reference and return CSV path."""
    counties = pd.DataFrame(
        [{"county_fips": 38017, "state_fips": 38, "county_name": "Cass County"}]
    )
    counties_path = tmp_path / "nd_counties.csv"
    counties.to_csv(counties_path, index=False)
    return counties_path


def _base_config(
    tmp_path: Path,
    projection_root: Path,
    crosswalk_path: Path,
    shares_path: Path,
    counties_path: Path,
    end_year: int,
) -> dict[str, object]:
    """Build minimal config for orchestrator tests."""
    return {
        "project": {"base_year": 2025, "projection_horizon": end_year - 2025},
        "output": {"compression": "gzip"},
        "pipeline": {"projection": {"output_dir": str(projection_root)}},
        "geography": {"reference_data": {"counties_file": str(counties_path)}},
        "place_projections": {
            "enabled": True,
            "crosswalk_path": str(crosswalk_path),
            "historical_shares_path": str(shares_path),
            "model": {
                "epsilon": 0.001,
                "lambda_decay": 0.9,
                "reconciliation_flag_threshold": 0.05,
                "history_start": 2020,
                "history_end": 2024,
            },
            "output": {"base_year": 2025, "end_year": end_year},
        },
    }


def test_hard_constraint_violation_raises_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IMP-13 hard constraints fail loudly when shares are out of bounds."""
    projection_root = tmp_path / "projections"
    county_dir = projection_root / "baseline" / "county"
    county_dir.mkdir(parents=True, exist_ok=True)
    county_df = _synthetic_county_cohorts(years=[2025], totals_by_year={2025: 1000.0})
    county_df.to_parquet(
        county_dir / "nd_county_38017_projection_2025_2025_baseline.parquet",
        index=False,
    )

    crosswalk_path = _minimal_crosswalk(tmp_path)
    shares_path = _minimal_share_history(tmp_path)
    counties_path = _minimal_counties_file(tmp_path)
    config = _base_config(
        tmp_path=tmp_path,
        projection_root=projection_root,
        crosswalk_path=crosswalk_path,
        shares_path=shares_path,
        counties_path=counties_path,
        end_year=2025,
    )

    def _invalid_trend(**_: object) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "county_fips": "38017",
                    "year": 2025,
                    "row_type": "place",
                    "place_fips": "3825700",
                    "projected_share_raw": 1.2,
                    "projected_share": 1.2,
                    "county_population": 1000.0,
                    "projected_population": 1200.0,
                    "base_share": 1.2,
                    "reconciliation_adjustment": 0.0,
                    "reconciliation_flag": False,
                    "fitting_method": "ols",
                    "constraint_method": "proportional",
                },
                {
                    "county_fips": "38017",
                    "year": 2025,
                    "row_type": "balance_of_county",
                    "place_fips": pd.NA,
                    "projected_share_raw": -0.2,
                    "projected_share": -0.2,
                    "county_population": 1000.0,
                    "projected_population": -200.0,
                    "base_share": -0.2,
                    "reconciliation_adjustment": 0.0,
                    "reconciliation_flag": False,
                    "fitting_method": "ols",
                    "constraint_method": "proportional",
                },
            ]
        )

    monkeypatch.setattr(orchestrator, "trend_all_places_in_county", _invalid_trend)
    with pytest.raises(ValueError, match="Share bound hard constraint failed"):
        orchestrator.run_place_projections(
            scenario="baseline",
            config=config,
            variant_winner={"fitting_method": "ols", "constraint_method": "proportional"},
        )


def test_soft_constraint_flags_do_not_block_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    IMP-13 soft constraints produce QA evidence without raising.

    This synthetic case intentionally includes:
    - tiny negative balance-of-county values (floating-point scale),
    - extreme annual growth beyond HIGH tier uncertainty band.
    """
    projection_root = tmp_path / "projections"
    county_dir = projection_root / "baseline" / "county"
    county_dir.mkdir(parents=True, exist_ok=True)
    county_df = _synthetic_county_cohorts(
        years=[2025, 2026],
        totals_by_year={2025: 1000.0, 2026: 3000.0},
    )
    county_df.to_parquet(
        county_dir / "nd_county_38017_projection_2025_2026_baseline.parquet",
        index=False,
    )

    crosswalk_path = _minimal_crosswalk(tmp_path)
    shares_path = _minimal_share_history(tmp_path)
    counties_path = _minimal_counties_file(tmp_path)
    config = _base_config(
        tmp_path=tmp_path,
        projection_root=projection_root,
        crosswalk_path=crosswalk_path,
        shares_path=shares_path,
        counties_path=counties_path,
        end_year=2026,
    )

    def _soft_flag_trend(**_: object) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "county_fips": "38017",
                    "year": 2025,
                    "row_type": "place",
                    "place_fips": "3825700",
                    "projected_share_raw": 1.0,
                    "projected_share": 1.0,
                    "county_population": 1000.0,
                    "projected_population": 1000.0000000005,
                    "base_share": 1.0,
                    "reconciliation_adjustment": 0.0,
                    "reconciliation_flag": False,
                    "fitting_method": "ols",
                    "constraint_method": "proportional",
                },
                {
                    "county_fips": "38017",
                    "year": 2025,
                    "row_type": "balance_of_county",
                    "place_fips": pd.NA,
                    "projected_share_raw": 0.0,
                    "projected_share": 0.0,
                    "county_population": 1000.0,
                    "projected_population": 0.0,
                    "base_share": 0.0,
                    "reconciliation_adjustment": 0.0,
                    "reconciliation_flag": False,
                    "fitting_method": "ols",
                    "constraint_method": "proportional",
                },
                {
                    "county_fips": "38017",
                    "year": 2026,
                    "row_type": "place",
                    "place_fips": "3825700",
                    "projected_share_raw": 1.0,
                    "projected_share": 1.0,
                    "county_population": 3000.0,
                    "projected_population": 3000.0000000005,
                    "base_share": 1.0,
                    "reconciliation_adjustment": 0.0,
                    "reconciliation_flag": False,
                    "fitting_method": "ols",
                    "constraint_method": "proportional",
                },
                {
                    "county_fips": "38017",
                    "year": 2026,
                    "row_type": "balance_of_county",
                    "place_fips": pd.NA,
                    "projected_share_raw": 0.0,
                    "projected_share": 0.0,
                    "county_population": 3000.0,
                    "projected_population": 0.0,
                    "base_share": 0.0,
                    "reconciliation_adjustment": 0.0,
                    "reconciliation_flag": False,
                    "fitting_method": "ols",
                    "constraint_method": "proportional",
                },
            ]
        )

    monkeypatch.setattr(orchestrator, "trend_all_places_in_county", _soft_flag_trend)
    result = orchestrator.run_place_projections(
        scenario="baseline",
        config=config,
        variant_winner={"fitting_method": "ols", "constraint_method": "proportional"},
    )

    assert result["places_processed"] == 1
    qa_dir = projection_root / "baseline" / "place" / "qa"
    outliers = pd.read_csv(qa_dir / "qa_outlier_flags.csv")
    assert "EXTREME_GROWTH" in set(outliers["flag_type"])

    balance = pd.read_csv(qa_dir / "qa_balance_of_county.csv")
    assert (balance["balance_of_county"] < 0.0).any()


def test_validate_state_scenario_ordering_raises_on_violation(tmp_path: Path) -> None:
    """Scenario ordering constraint raises when restricted > baseline or baseline > high."""
    projection_root = tmp_path / "projections"
    for scenario, total in [
        ("restricted_growth", 1200.0),
        ("baseline", 1100.0),
        ("high_growth", 1000.0),
    ]:
        county_dir = projection_root / scenario / "county"
        county_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([{"year": 2025, "population": total}])
        df.to_parquet(
            county_dir / f"nd_county_38017_projection_2025_2025_{scenario}.parquet",
            index=False,
        )

    config = {"pipeline": {"projection": {"output_dir": str(projection_root)}}}
    with pytest.raises(ValueError, match="Scenario ordering hard constraint failed"):
        orchestrator.validate_state_scenario_ordering(
            config=config,
            base_year=2025,
            end_year=2025,
            skip_if_missing=False,
        )
