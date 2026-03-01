"""
Tests for PP-003 IMP-05 place share-trending model utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cohort_projections.data.process.place_share_trending import (
    BALANCE_KEY,
    apply_cap_and_redistribute,
    apply_proportional_rescaling,
    compute_recency_weights,
    fit_share_trend,
    inverse_logit,
    logit_transform,
    project_shares,
    reconcile_county_shares,
    trend_all_places_in_county,
)


def _make_history_with_balance(
    county_fips: str,
    place_shares_by_year: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """Build synthetic county place-share history including balance rows."""
    rows: list[dict[str, object]] = []
    for year, place_map in sorted(place_shares_by_year.items()):
        place_sum = 0.0
        for place_fips, share in place_map.items():
            place_sum += share
            rows.append(
                {
                    "county_fips": county_fips,
                    "year": year,
                    "row_type": "place",
                    "place_fips": place_fips,
                    "share_raw": share,
                }
            )
        rows.append(
            {
                "county_fips": county_fips,
                "year": year,
                "row_type": BALANCE_KEY,
                "place_fips": pd.NA,
                "share_raw": 1.0 - place_sum,
            }
        )
    return pd.DataFrame(rows)


def test_logit_transform_known_values_and_epsilon_clamping() -> None:
    """Logit transform handles boundaries via epsilon clamping."""
    shares = np.array([0.0, 0.5, 1.0])
    transformed = logit_transform(shares, epsilon=0.001)

    expected = np.array(
        [
            np.log(0.001 / 0.999),
            0.0,
            np.log(0.999 / 0.001),
        ]
    )
    np.testing.assert_allclose(transformed, expected, rtol=1e-12, atol=1e-12)


def test_inverse_logit_round_trip_matches_clamped_shares() -> None:
    """inverse_logit(logit_transform(s)) ~= clamp(s)."""
    shares = np.array([0.0, 0.02, 0.5, 0.97, 1.0])
    round_trip = inverse_logit(logit_transform(shares, epsilon=0.001))
    expected = np.clip(shares, 0.001, 0.999)
    np.testing.assert_allclose(round_trip, expected, rtol=1e-12, atol=1e-12)


def test_fit_share_trend_ols_recovers_known_coefficients() -> None:
    """OLS fit recovers coefficients for a noiseless centered-linear series."""
    years = np.arange(2000, 2005, dtype=float)
    center_year = float(np.mean(years))
    expected_intercept = -1.25
    expected_slope = 0.2

    logit_shares = expected_intercept + expected_slope * (years - center_year)
    intercept, slope = fit_share_trend(logit_shares, years, method="ols")

    np.testing.assert_allclose(intercept, expected_intercept, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(slope, expected_slope, rtol=1e-12, atol=1e-12)


def test_fit_share_trend_wls_uses_expected_lambda_weights() -> None:
    """WLS uses lambda-decay weights and matches manual weighted least squares."""
    years = np.array([2000, 2001, 2002, 2003, 2004], dtype=float)
    expected_weights = np.array([0.9**4, 0.9**3, 0.9**2, 0.9, 1.0], dtype=float)
    np.testing.assert_allclose(
        compute_recency_weights(years, lambda_decay=0.9),
        expected_weights,
        rtol=1e-12,
        atol=1e-12,
    )

    logit_shares = np.array([-1.0, -0.6, -0.3, 0.4, 0.9], dtype=float)
    intercept, slope = fit_share_trend(logit_shares, years, method="wls", lambda_decay=0.9)

    centered_years = years - np.mean(years)
    design = np.column_stack([np.ones(len(years)), centered_years])
    sqrt_w = np.sqrt(expected_weights)
    manual_coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], logit_shares * sqrt_w, rcond=None)

    np.testing.assert_allclose([intercept, slope], manual_coef, rtol=1e-12, atol=1e-12)


def test_time_centering_parameterization_is_projection_invariant() -> None:
    """Projected shares are invariant under equivalent centered/uncentered forms."""
    years = np.array([2000, 2001, 2002, 2003, 2004], dtype=float)
    logit_shares = np.array([-0.8, -0.6, -0.4, -0.2, 0.0], dtype=float)
    projection_years = np.array([2025, 2030], dtype=float)

    intercept_centered, slope = fit_share_trend(logit_shares, years, method="ols")
    center_year = float(np.mean(years))
    centered_projection = project_shares(
        intercept_centered,
        slope,
        projection_years=projection_years,
        center_year=center_year,
    )

    intercept_uncentered = intercept_centered - slope * center_year
    uncentered_projection = project_shares(
        intercept_uncentered,
        slope,
        projection_years=projection_years,
        center_year=0.0,
    )
    np.testing.assert_allclose(centered_projection, uncentered_projection, rtol=1e-12, atol=1e-12)


def test_apply_proportional_rescaling_sums_to_one_and_preserves_proportions() -> None:
    """Proportional rescaling preserves pairwise proportions and normalizes total."""
    shares = {"a": 0.8, "b": 0.4, "c": 0.3}
    adjusted = apply_proportional_rescaling(shares)

    np.testing.assert_allclose(sum(adjusted.values()), 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        adjusted["a"] / adjusted["b"],
        shares["a"] / shares["b"],
        rtol=1e-12,
        atol=1e-12,
    )


def test_cap_and_redistribute_keeps_declining_shares_unchanged() -> None:
    """Declining/stable places remain unchanged while growing places absorb excess."""
    projected = {"declining": 0.50, "growing": 0.40, "stable": 0.20}
    base = {"declining": 0.55, "growing": 0.25, "stable": 0.20}

    adjusted = apply_cap_and_redistribute(projected, base)

    np.testing.assert_allclose(adjusted["declining"], projected["declining"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(adjusted["stable"], projected["stable"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(adjusted["growing"], 0.30, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sum(adjusted.values()), 1.0, rtol=1e-12, atol=1e-12)


def test_cap_and_redistribute_iterative_clamping_handles_over_correction() -> None:
    """Iterative clamping prevents growing places from dropping below base shares."""
    projected = {"a": 0.65, "b": 0.31, "c": 0.20}
    base = {"a": 0.50, "b": 0.30, "c": 0.20}

    adjusted = apply_cap_and_redistribute(projected, base)

    np.testing.assert_allclose(adjusted["b"], base["b"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(adjusted["a"], base["a"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(adjusted["c"], projected["c"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sum(adjusted.values()), 1.0, rtol=1e-12, atol=1e-12)


def test_cap_and_redistribute_falls_back_when_all_places_grew() -> None:
    """When all places grew, cap-and-redistribute reduces to proportional rescaling."""
    projected = {"a": 0.45, "b": 0.70}
    base = {"a": 0.40, "b": 0.60}

    adjusted_cap = apply_cap_and_redistribute(projected, base)
    adjusted_prop = apply_proportional_rescaling(projected)

    np.testing.assert_allclose(
        [adjusted_cap["a"], adjusted_cap["b"]],
        [adjusted_prop["a"], adjusted_prop["b"]],
        rtol=1e-12,
        atol=1e-12,
    )


def test_reconcile_county_shares_includes_balance_and_flags_large_adjustment() -> None:
    """Reconciliation treats balance as a component and flags large discrepancies."""
    place_shares = {"a": 0.70, "b": 0.40}
    balance_share = 0.05
    base_shares = {"a": 0.65, "b": 0.30, BALANCE_KEY: 0.05}

    result = reconcile_county_shares(
        place_shares=place_shares,
        balance_share=balance_share,
        constraint_method="proportional",
        base_shares=base_shares,
        flag_threshold=0.05,
    )

    assert result.flagged
    np.testing.assert_allclose(
        sum(result.place_shares.values()) + result.balance_share,
        1.0,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(result.balance_share, balance_share / 1.15, rtol=1e-12, atol=1e-12)


def test_reconcile_county_shares_single_place_county_is_near_trivial() -> None:
    """Single-place counties with near-identity totals require negligible adjustment."""
    result = reconcile_county_shares(
        place_shares={"3825700": 0.719999999995},
        balance_share=0.280000000004,
        constraint_method="cap_and_redistribute",
        base_shares={"3825700": 0.72, BALANCE_KEY: 0.28},
    )

    assert result.adjustment < 1e-9
    assert not result.flagged
    np.testing.assert_allclose(result.place_shares["3825700"], 0.719999999995, rtol=1e-12, atol=1e-12)


def test_trend_all_places_in_county_with_no_projected_places_returns_empty() -> None:
    """Counties with no projected places return no share rows."""
    empty_history = pd.DataFrame(columns=["county_fips", "year", "row_type", "place_fips", "share_raw"])
    county_pop = pd.DataFrame({"year": [2025], "county_population": [1000]})

    result = trend_all_places_in_county(
        place_share_history=empty_history,
        county_pop_history=county_pop,
        config={"projection_years": [2025]},
    )

    assert result.empty


def test_trend_all_places_sets_population_zero_for_near_epsilon_share() -> None:
    """Projected shares below epsilon are converted to zero population."""
    place_history = _make_history_with_balance(
        county_fips="38089",
        place_shares_by_year={
            2022: {"3899999": 0.0100},
            2023: {"3899999": 0.0030},
            2024: {"3899999": 0.0010},
        },
    )
    county_pop = pd.DataFrame({"year": [2025], "county_population": [1000]})

    result = trend_all_places_in_county(
        place_share_history=place_history,
        county_pop_history=county_pop,
        config={
            "projection_years": [2025],
            "epsilon": 0.001,
            "fitting_method": "ols",
            "constraint_method": "proportional",
        },
    )

    place_row = result[(result["year"] == 2025) & (result["row_type"] == "place")].iloc[0]
    assert place_row["projected_share"] < 0.001
    assert place_row["projected_population"] == 0.0


def test_trend_all_places_end_to_end_synthetic_county_reconciles_to_one() -> None:
    """End-to-end synthetic county projections sum to 1.0 for each year."""
    place_history = _make_history_with_balance(
        county_fips="38017",
        place_shares_by_year={
            2020: {"3825700": 0.45, "3884780": 0.25, "3890000": 0.10},
            2021: {"3825700": 0.46, "3884780": 0.24, "3890000": 0.10},
            2022: {"3825700": 0.47, "3884780": 0.23, "3890000": 0.10},
            2023: {"3825700": 0.48, "3884780": 0.22, "3890000": 0.10},
            2024: {"3825700": 0.49, "3884780": 0.21, "3890000": 0.10},
        },
    )
    county_pop = pd.DataFrame(
        {
            "year": [2025, 2026, 2027],
            "county_population": [130_000, 131_000, 132_000],
        }
    )

    result = trend_all_places_in_county(
        place_share_history=place_history,
        county_pop_history=county_pop,
        config={
            "projection_years": [2025, 2026, 2027],
            "epsilon": 0.001,
            "lambda_decay": 0.9,
            "fitting_method": "wls",
            "constraint_method": "cap_and_redistribute",
            "reconciliation_flag_threshold": 0.05,
        },
    )

    assert set(result["year"]) == {2025, 2026, 2027}
    for _, year_group in result.groupby("year"):
        np.testing.assert_allclose(
            year_group["projected_share"].sum(),
            1.0,
            rtol=1e-10,
            atol=1e-10,
        )

    assert len(result[result["row_type"] == "place"]) == 9
    assert len(result[result["row_type"] == BALANCE_KEY]) == 3
