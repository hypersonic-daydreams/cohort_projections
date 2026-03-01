"""
Tests for PP-003 IMP-12 QA artifact generation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cohort_projections.data.process.place_projection_orchestrator import (
    ALLOWED_OUTLIER_FLAG_TYPES,
    run_place_projections,
)


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


def test_run_place_projections_writes_qa_artifacts_with_contract_schema(tmp_path: Path) -> None:
    """IMP-12: All four S06 QA artifacts are written with expected columns."""
    projection_root = tmp_path / "projections"
    county_dir = projection_root / "baseline" / "county"
    county_dir.mkdir(parents=True, exist_ok=True)

    county_df = _synthetic_county_cohorts(
        years=[2025, 2026, 2027],
        totals_by_year={2025: 1000.0, 2026: 1600.0, 2027: 2400.0},
    )
    county_path = county_dir / "nd_county_38017_projection_2025_2027_baseline.parquet"
    county_df.to_parquet(county_path, index=False)

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
            },
            {
                "state_fips": 38,
                "place_fips": 3884780,
                "place_name": "Valley city",
                "county_fips": 38017,
                "assignment_type": "single_county",
                "area_share": 1.0,
                "historical_only": False,
                "source_vintage": 2020,
                "source_method": "test",
                "confidence_tier": "MODERATE",
                "tier_boundary": False,
            },
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
            },
        ]
    )
    crosswalk_path = tmp_path / "place_county_crosswalk_2020.csv"
    crosswalk.to_csv(crosswalk_path, index=False)

    share_rows: list[dict[str, object]] = []
    for year, high, moderate, lower in [
        (2020, 0.62, 0.20, 0.06),
        (2021, 0.60, 0.20, 0.06),
        (2022, 0.58, 0.20, 0.06),
        (2023, 0.56, 0.20, 0.06),
        (2024, 0.54, 0.20, 0.06),
    ]:
        share_rows.extend(
            [
                {
                    "county_fips": "38017",
                    "place_fips": "3825700",
                    "year": year,
                    "row_type": "place",
                    "share_raw": high,
                },
                {
                    "county_fips": "38017",
                    "place_fips": "3884780",
                    "year": year,
                    "row_type": "place",
                    "share_raw": moderate,
                },
                {
                    "county_fips": "38017",
                    "place_fips": "3890000",
                    "year": year,
                    "row_type": "place",
                    "share_raw": lower,
                },
            ]
        )
    shares_path = tmp_path / "place_shares_2000_2024.parquet"
    pd.DataFrame(share_rows).to_parquet(shares_path, index=False)

    counties = pd.DataFrame(
        [{"county_fips": 38017, "state_fips": 38, "county_name": "Cass County"}]
    )
    counties_path = tmp_path / "nd_counties.csv"
    counties.to_csv(counties_path, index=False)

    config = {
        "project": {"base_year": 2025, "projection_horizon": 2},
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
            "output": {"base_year": 2025, "end_year": 2027},
        },
    }

    run_place_projections(
        scenario="baseline",
        config=config,
        variant_winner={"fitting_method": "wls", "constraint_method": "cap_and_redistribute"},
    )

    qa_dir = projection_root / "baseline" / "place" / "qa"
    assert qa_dir.exists()

    tier_summary = pd.read_csv(qa_dir / "qa_tier_summary.csv")
    assert list(tier_summary.columns) == [
        "confidence_tier",
        "place_count",
        "total_base_population",
        "total_final_population",
        "mean_growth_rate",
        "median_growth_rate",
        "min_growth_rate",
        "max_growth_rate",
    ]
    assert list(tier_summary["confidence_tier"]) == ["HIGH", "MODERATE", "LOWER"]
    assert len(tier_summary) == 3

    share_sum = pd.read_csv(qa_dir / "qa_share_sum_validation.csv")
    assert list(share_sum.columns) == [
        "county_fips",
        "county_name",
        "year",
        "sum_place_shares",
        "balance_of_county_share",
        "constraint_satisfied",
        "rescaling_applied",
    ]
    assert len(share_sum) == 3
    assert share_sum["constraint_satisfied"].astype(bool).all()

    outlier_flags = pd.read_csv(qa_dir / "qa_outlier_flags.csv")
    assert list(outlier_flags.columns) == [
        "place_fips",
        "name",
        "confidence_tier",
        "flag_type",
        "flag_detail",
        "year",
    ]
    assert not outlier_flags.empty
    assert set(outlier_flags["flag_type"]).issubset(ALLOWED_OUTLIER_FLAG_TYPES)
    assert "EXTREME_GROWTH" in set(outlier_flags["flag_type"])

    balance = pd.read_csv(qa_dir / "qa_balance_of_county.csv")
    assert list(balance.columns) == [
        "county_fips",
        "county_name",
        "year",
        "county_total",
        "sum_of_places",
        "balance_of_county",
        "balance_share",
    ]
    assert len(balance) == 3
