"""
Tests for PP-003 IMP-06 place projection orchestrator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from cohort_projections.data.process.place_projection_orchestrator import (
    allocate_age_sex_detail,
    run_place_projections,
)


def _synthetic_county_cohorts(years: list[int], totals_by_year: dict[int, float]) -> pd.DataFrame:
    """Build synthetic county cohorts with single-year age, sex, and race columns."""
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
                        "race": "White alone, Non-Hispanic",
                        "population": cell_value,
                    }
                )
    return pd.DataFrame(rows)


def test_allocate_age_sex_detail_high_outputs_18x2_and_sum_matches_total() -> None:
    """HIGH tier allocation yields 18 age groups x 2 sexes and sums to place total."""
    county_df = _synthetic_county_cohorts(years=[2025], totals_by_year={2025: 1000.0})

    allocated = allocate_age_sex_detail(
        place_total=250.0,
        county_cohort_df=county_df[county_df["year"] == 2025],
        tier="HIGH",
    )

    assert list(allocated.columns) == ["age_group", "sex", "population"]
    assert len(allocated) == 36
    assert set(allocated["sex"]) == {"Male", "Female"}
    assert len(set(allocated["age_group"])) == 18
    np.testing.assert_allclose(allocated["population"].sum(), 250.0, rtol=1e-9, atol=1e-9)


def test_allocate_age_sex_detail_moderate_outputs_6x2_and_sum_matches_total() -> None:
    """MODERATE tier allocation yields 6 age groups x 2 sexes and sums to place total."""
    county_df = _synthetic_county_cohorts(years=[2025], totals_by_year={2025: 900.0})

    allocated = allocate_age_sex_detail(
        place_total=90.0,
        county_cohort_df=county_df[county_df["year"] == 2025],
        tier="MODERATE",
    )

    assert list(allocated.columns) == ["age_group", "sex", "population"]
    assert len(allocated) == 12
    assert set(allocated["sex"]) == {"Male", "Female"}
    assert set(allocated["age_group"]) == {"0-17", "18-24", "25-44", "45-64", "65-84", "85+"}
    np.testing.assert_allclose(allocated["population"].sum(), 90.0, rtol=1e-9, atol=1e-9)


def test_allocate_age_sex_detail_lower_returns_total_only() -> None:
    """LOWER tier returns one total-population row."""
    county_df = _synthetic_county_cohorts(years=[2025], totals_by_year={2025: 750.0})

    allocated = allocate_age_sex_detail(
        place_total=30.0,
        county_cohort_df=county_df[county_df["year"] == 2025],
        tier="LOWER",
    )

    assert list(allocated.columns) == ["population"]
    assert len(allocated) == 1
    np.testing.assert_allclose(float(allocated["population"].iloc[0]), 30.0, rtol=1e-9, atol=1e-9)


def test_run_place_projections_writes_contract_outputs(tmp_path: Path) -> None:
    """
    End-to-end synthetic run writes place outputs matching the S06 contract.

    Validates:
    - Tier-specific output schemas
    - Required metadata JSON fields
    - Aggregate summary with balance-of-county row
    - Parquet footer key-value metadata
    - Naming and directory structure
    """
    projection_root = tmp_path / "projections"
    county_dir = projection_root / "baseline" / "county"
    county_dir.mkdir(parents=True, exist_ok=True)

    county_df = _synthetic_county_cohorts(
        years=[2025, 2026],
        totals_by_year={2025: 1000.0, 2026: 1100.0},
    )
    county_parquet = county_dir / "nd_county_38017_projection_2025_2026_baseline.parquet"
    county_df.to_parquet(county_parquet, index=False)

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
        (2020, 0.40, 0.12, 0.030),
        (2021, 0.41, 0.12, 0.029),
        (2022, 0.42, 0.11, 0.028),
        (2023, 0.43, 0.11, 0.027),
        (2024, 0.44, 0.10, 0.026),
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
        "project": {"base_year": 2025, "projection_horizon": 1},
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
            "output": {"base_year": 2025, "end_year": 2026},
        },
    }

    result = run_place_projections(
        scenario="baseline",
        config=config,
        variant_winner={"fitting_method": "ols", "constraint_method": "proportional"},
    )

    assert result["places_processed"] == 3
    assert result["balance_rows"] == 1

    place_dir = projection_root / "baseline" / "place"
    assert place_dir.exists()

    expected_place_files = {
        "3825700": "HIGH",
        "3884780": "MODERATE",
        "3890000": "LOWER",
    }
    required_footer_keys = {
        b"scenario",
        b"geography_level",
        b"place_fips",
        b"county_fips",
        b"confidence_tier",
        b"projection_base_year",
        b"projection_end_year",
        b"model_method",
        b"model_version",
        b"crosswalk_vintage",
        b"processing_date",
    }

    for place_fips, tier in expected_place_files.items():
        stem = f"nd_place_{place_fips}_projection_2025_2026_baseline"
        parquet_path = place_dir / f"{stem}.parquet"
        metadata_path = place_dir / f"{stem}_metadata.json"
        summary_path = place_dir / f"{stem}_summary.csv"

        assert parquet_path.exists()
        assert metadata_path.exists()
        assert summary_path.exists()

        parquet_df = pd.read_parquet(parquet_path)
        if tier in {"HIGH", "MODERATE"}:
            assert list(parquet_df.columns) == ["year", "age_group", "sex", "population"]
            rows_per_year = parquet_df.groupby("year").size()
            expected_rows = 36 if tier == "HIGH" else 12
            assert set(rows_per_year.tolist()) == {expected_rows}
        else:
            assert list(parquet_df.columns) == ["year", "population"]
            assert len(parquet_df) == 2

        with open(metadata_path, encoding="utf-8") as file_handle:
            metadata = json.load(file_handle)
        assert set(metadata.keys()) == {
            "geography",
            "projection",
            "share_model",
            "summary_statistics",
            "validation",
            "processing_time_seconds",
        }
        assert metadata["geography"]["place_fips"] == place_fips
        assert metadata["projection"]["scenario"] == "baseline"

        parquet_meta = pq.read_metadata(parquet_path).metadata
        assert required_footer_keys.issubset(set(parquet_meta.keys()))

    places_summary = pd.read_csv(place_dir / "places_summary.csv")
    assert list(places_summary.columns) == [
        "place_fips",
        "name",
        "county_fips",
        "level",
        "row_type",
        "confidence_tier",
        "base_population",
        "final_population",
        "absolute_growth",
        "growth_rate",
        "base_share",
        "final_share",
        "processing_time",
    ]
    assert "balance_of_county" in set(places_summary["row_type"])
    assert len(places_summary) == 4

    with open(place_dir / "places_metadata.json", encoding="utf-8") as file_handle:
        run_meta = json.load(file_handle)
    assert run_meta["level"] == "place"
    assert run_meta["num_geographies"] == 3
    assert run_meta["by_tier"] == {"HIGH": 1, "MODERATE": 1, "LOWER": 1}

