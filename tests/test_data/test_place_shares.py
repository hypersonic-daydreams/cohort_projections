"""
Tests for historical place share computation (PP-003 IMP-03).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cohort_projections.data.process.place_shares import (
    compute_historical_shares,
    normalize_county_population_history,
)


def _place_history_fixture() -> pd.DataFrame:
    """Synthetic place history with two counties and one dissolved place."""
    rows = [
        # Cass County 38017
        {"place_fips": "3825700", "place_name": "Fargo city", "county_fips": "38017", "year": 2023, "population": 80_000},
        {"place_fips": "3884780", "place_name": "West Fargo city", "county_fips": "38017", "year": 2023, "population": 30_000},
        {"place_fips": "3825700", "place_name": "Fargo city", "county_fips": "38017", "year": 2024, "population": 82_000},
        {"place_fips": "3884780", "place_name": "West Fargo city", "county_fips": "38017", "year": 2024, "population": 31_000},
        # Burleigh County 38015
        {"place_fips": "3807200", "place_name": "Bismarck city", "county_fips": "38015", "year": 2023, "population": 75_000},
        {"place_fips": "3849900", "place_name": "Mandan city", "county_fips": "38015", "year": 2023, "population": 24_000},
        {"place_fips": "3807200", "place_name": "Bismarck city", "county_fips": "38015", "year": 2024, "population": 76_000},
        {"place_fips": "3849900", "place_name": "Mandan city", "county_fips": "38015", "year": 2024, "population": 25_000},
    ]
    return pd.DataFrame(rows)


def _county_history_wide_fixture() -> pd.DataFrame:
    """Synthetic county totals in wide PEP-like schema."""
    return pd.DataFrame(
        [
            {
                "STATE": "38",
                "COUNTY": "017",
                "POPESTIMATE2023": 130_000,
                "POPESTIMATE2024": 132_000,
            },
            {
                "STATE": "38",
                "COUNTY": "015",
                "POPESTIMATE2023": 120_000,
                "POPESTIMATE2024": 122_000,
            },
            # County with no places in fixture (should not appear in output rows)
            {
                "STATE": "38",
                "COUNTY": "089",
                "POPESTIMATE2023": 33_000,
                "POPESTIMATE2024": 33_500,
            },
        ],
    )


def test_normalize_county_population_history_handles_wide_schema() -> None:
    """Wide PEP-style county totals are melted into long county/year rows."""
    long = normalize_county_population_history(
        county_population_history=_county_history_wide_fixture(),
        year_min=2023,
        year_max=2024,
    )

    assert set(long.columns) == {"county_fips", "year", "county_population"}
    assert len(long) == 6  # 3 counties x 2 years
    assert long["county_fips"].str.fullmatch(r"\d{5}").all()
    assert set(long["year"]) == {2023, 2024}


def test_compute_historical_shares_bounds_and_balance_invariant() -> None:
    """Shares are bounded and place+balance shares sum to 1 by county-year."""
    shares = compute_historical_shares(
        place_population_history=_place_history_fixture(),
        county_population_history=_county_history_wide_fixture(),
        epsilon=0.001,
        year_min=2023,
        year_max=2024,
    )

    place_rows = shares[shares["row_type"] == "place"].copy()
    balance_rows = shares[shares["row_type"] == "balance_of_county"].copy()

    assert not place_rows.empty
    assert not balance_rows.empty

    # Raw shares should be within [0, 1].
    assert (place_rows["share_raw"] >= 0).all()
    assert (place_rows["share_raw"] <= 1).all()

    # Clamped shares should be in [0.001, 0.999] for all rows.
    assert (shares["share"] >= 0.001).all()
    assert (shares["share"] <= 0.999).all()

    # Balance shares are non-negative.
    assert (balance_rows["share_raw"] >= 0).all()

    # Place sum + balance = 1.0 by county-year using raw (unclamped) shares.
    place_sum = (
        place_rows.groupby(["county_fips", "year"], as_index=False)["share_raw"]
        .sum()
        .rename(columns={"share_raw": "place_share_sum"})
    )
    balance_sum = (
        balance_rows.groupby(["county_fips", "year"], as_index=False)["share_raw"]
        .sum()
        .rename(columns={"share_raw": "balance_share_sum"})
    )
    check = place_sum.merge(balance_sum, on=["county_fips", "year"], how="inner", validate="one_to_one")
    np.testing.assert_allclose(
        check["place_share_sum"] + check["balance_share_sum"],
        np.ones(len(check)),
        rtol=1e-10,
        atol=1e-10,
    )


def test_counties_without_place_rows_do_not_emit_share_rows() -> None:
    """A county present only in county totals appears nowhere in output."""
    shares = compute_historical_shares(
        place_population_history=_place_history_fixture(),
        county_population_history=_county_history_wide_fixture(),
        epsilon=0.001,
        year_min=2023,
        year_max=2024,
    )
    assert "38089" not in set(shares["county_fips"])


def test_fargo_share_of_cass_is_plausible() -> None:
    """Fargo share in Cass is within plausible range (roughly 0.6-0.7)."""
    shares = compute_historical_shares(
        place_population_history=_place_history_fixture(),
        county_population_history=_county_history_wide_fixture(),
        epsilon=0.001,
        year_min=2024,
        year_max=2024,
    )
    fargo = shares[
        (shares["row_type"] == "place")
        & (shares["place_fips"] == "3825700")
        & (shares["county_fips"] == "38017")
        & (shares["year"] == 2024)
    ].iloc[0]
    assert 0.60 <= fargo["share_raw"] <= 0.70
