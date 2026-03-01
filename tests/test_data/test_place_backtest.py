"""
Tests for PP-003 IMP-08 place backtesting utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cohort_projections.data.process.place_backtest import (
    compute_per_place_metrics,
    compute_tier_aggregates,
    compute_variant_score,
    run_single_variant,
    select_winner,
)


def _synthetic_share_history() -> pd.DataFrame:
    """Create synthetic county share history with place populations."""
    rows: list[dict[str, object]] = []
    county_pop_by_year = {2000: 1000.0, 2001: 1000.0, 2002: 1000.0, 2003: 1000.0, 2004: 1000.0}
    shares = {
        2000: {"3825700": 0.40, "3884780": 0.30},
        2001: {"3825700": 0.41, "3884780": 0.29},
        2002: {"3825700": 0.42, "3884780": 0.28},
        2003: {"3825700": 0.43, "3884780": 0.27},
        2004: {"3825700": 0.44, "3884780": 0.26},
    }

    for year, place_shares in shares.items():
        for place_fips, share in place_shares.items():
            county_pop = county_pop_by_year[year]
            rows.append(
                {
                    "county_fips": "38017",
                    "place_fips": place_fips,
                    "place_name": "Synthetic Place",
                    "year": year,
                    "row_type": "place",
                    "share_raw": share,
                    "county_population": county_pop,
                    "population": share * county_pop,
                }
            )
    return pd.DataFrame(rows)


def test_compute_per_place_metrics_matches_known_values() -> None:
    """Per-place metrics match known synthetic calculations."""
    projected = pd.DataFrame(
        [
            {"place_fips": "3825700", "year": 2023, "projected_population": 110.0},
            {"place_fips": "3825700", "year": 2024, "projected_population": 108.0},
            {"place_fips": "3884780", "year": 2023, "projected_population": 45.0},
            {"place_fips": "3884780", "year": 2024, "projected_population": 44.0},
        ]
    )
    actual = pd.DataFrame(
        [
            {"place_fips": "3825700", "year": 2023, "actual_population": 100.0},
            {"place_fips": "3825700", "year": 2024, "actual_population": 120.0},
            {"place_fips": "3884780", "year": 2023, "actual_population": 50.0},
            {"place_fips": "3884780", "year": 2024, "actual_population": 40.0},
        ]
    )

    metrics = compute_per_place_metrics(projected, actual).set_index("place_fips")
    np.testing.assert_allclose(metrics.loc["3825700", "MAPE"], 10.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(metrics.loc["3825700", "MedAPE"], 10.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(metrics.loc["3825700", "ME"], 0.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(metrics.loc["3825700", "MaxAPE"], 10.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(metrics.loc["3825700", "AE_terminal"], 12.0, rtol=1e-12, atol=1e-12)


def test_compute_tier_aggregates_uses_median_for_tier_medape() -> None:
    """Tier MedAPE uses median of place MAPE values (not mean)."""
    place_metrics = pd.DataFrame(
        [
            {"place_fips": "3800001", "MAPE": 5.0, "ME": 1.0},
            {"place_fips": "3800002", "MAPE": 50.0, "ME": 2.0},
            {"place_fips": "3800003", "MAPE": 100.0, "ME": 3.0},
        ]
    )
    tiers = pd.DataFrame(
        [
            {"place_fips": "3800001", "confidence_tier": "HIGH", "population_2024": 1000},
            {"place_fips": "3800002", "confidence_tier": "HIGH", "population_2024": 1000},
            {"place_fips": "3800003", "confidence_tier": "HIGH", "population_2024": 1000},
        ]
    )

    aggregated = compute_tier_aggregates(place_metrics, tiers).set_index("confidence_tier")
    np.testing.assert_allclose(aggregated.loc["HIGH", "tier_medape"], 50.0, rtol=1e-12, atol=1e-12)


def test_compute_variant_score_population_weighted_formula_matches_spec() -> None:
    """Population-weighted MedAPE score matches S04 Section 5.3 formula."""
    tier_aggregates = pd.DataFrame(
        [
            {"confidence_tier": "HIGH", "tier_medape": 10.0, "tier_population_2024": 1000.0},
            {"confidence_tier": "MODERATE", "tier_medape": 20.0, "tier_population_2024": 500.0},
            {"confidence_tier": "LOWER", "tier_medape": 40.0, "tier_population_2024": 500.0},
        ]
    )
    score = compute_variant_score(tier_aggregates)
    np.testing.assert_allclose(score, 20.0, rtol=1e-12, atol=1e-12)


def test_compute_variant_score_ignores_excluded_tier() -> None:
    """EXCLUDED tier contributes no weight to winner score."""
    tier_aggregates = pd.DataFrame(
        [
            {"confidence_tier": "HIGH", "tier_medape": 10.0, "tier_population_2024": 100.0},
            {"confidence_tier": "MODERATE", "tier_medape": 20.0, "tier_population_2024": 100.0},
            {"confidence_tier": "LOWER", "tier_medape": 30.0, "tier_population_2024": 100.0},
            {"confidence_tier": "EXCLUDED", "tier_medape": 1.0, "tier_population_2024": 100000.0},
        ]
    )
    score = compute_variant_score(tier_aggregates)
    np.testing.assert_allclose(score, 20.0, rtol=1e-12, atol=1e-12)


def test_select_winner_tie_break_prefers_a_over_b() -> None:
    """Tie at 3 decimals prefers A variants before B variants."""
    winner = select_winner({"A-II": 1.2344, "B-I": 1.2344})
    assert winner == "A-II"


def test_select_winner_tie_break_prefers_i_over_ii_within_fitting_family() -> None:
    """Tie at 3 decimals prefers I before II within same fitting family."""
    winner = select_winner({"A-I": 1.2344, "A-II": 1.2344})
    assert winner == "A-I"


def test_select_winner_returns_single_variant_id() -> None:
    """Winner selection returns exactly one variant identifier."""
    winner = select_winner({"A-I": 2.0, "A-II": 1.9, "B-II": 1.8})
    assert winner == "B-II"
    assert isinstance(winner, str)


def test_run_single_variant_executes_all_four_variants_on_synthetic_data() -> None:
    """All 2x2 matrix variants execute and emit projected/actual rows."""
    share_history = _synthetic_share_history()
    county_pop = pd.DataFrame(
        [
            {"county_fips": "38017", "year": 2003, "county_population": 1000.0},
            {"county_fips": "38017", "year": 2004, "county_population": 1000.0},
        ]
    )
    config = {"place_projections": {"model": {"epsilon": 0.001, "lambda_decay": 0.9}}}

    variants = [
        ("A-I", "ols", "proportional"),
        ("A-II", "ols", "cap_and_redistribute"),
        ("B-I", "wls", "proportional"),
        ("B-II", "wls", "cap_and_redistribute"),
    ]

    for variant_id, fitting_method, constraint_method in variants:
        result = run_single_variant(
            variant_id=variant_id,
            fitting_method=fitting_method,
            constraint_method=constraint_method,
            train_years=[2000, 2002],
            test_years=[2003, 2004],
            share_history=share_history,
            county_pop=county_pop,
            config=config,
        )
        projected = result["projected"]
        actual = result["actual"]

        assert not projected.empty
        assert not actual.empty
        assert set(projected["year"].unique()) == {2003, 2004}
        assert set(actual["year"].unique()) == {2003, 2004}
        assert projected["projected_population"].notna().all()

