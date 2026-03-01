"""
Tests for place population history assembly (PP-003 IMP-02 + IMP-04 wiring).
"""

from __future__ import annotations

import pandas as pd

from scripts.data.assemble_place_population_history import (
    assemble_place_population_history,
    attach_crosswalk,
    build_tiers_from_history,
    validate_place_history,
)


def _make_place_vintage(
    years: list[int],
    include_dissolved: bool,
    source_tag: str,
) -> pd.DataFrame:
    """Create a synthetic PEP place vintage in wide format."""
    rows: list[dict[str, object]] = []
    places = [
        ("25700", "Fargo city", 110_000),  # active
        ("07200", "Bismarck city", 74_000),  # active
    ]
    if include_dissolved:
        places.append(("04740", "Bantry city", 12))  # dissolved historical-only

    for place_code, place_name, base_pop in places:
        record: dict[str, object] = {
            "SUMLEV": "162",
            "STATE": "38",
            "PLACE": place_code,
            "NAME": place_name,
            "FUNCSTAT": "A",
            "SOURCE": source_tag,
        }
        for offset, year in enumerate(years):
            record[f"POPESTIMATE{year}"] = base_pop + offset
        rows.append(record)

    return pd.DataFrame(rows)


def _crosswalk_fixture() -> pd.DataFrame:
    """Synthetic crosswalk with two active places and one dissolved place."""
    return pd.DataFrame(
        [
            {
                "place_fips": "3825700",
                "county_fips": "38017",
                "historical_only": False,
            },
            {
                "place_fips": "3807200",
                "county_fips": "38015",
                "historical_only": False,
            },
            {
                "place_fips": "3804740",
                "county_fips": "38049",
                "historical_only": True,
            },
        ],
    )


def test_assemble_history_handoff_windows_and_schema() -> None:
    """History uses the required S02 windows and preserves vintage provenance."""
    sub_est00int = _make_place_vintage(list(range(2000, 2011)), include_dissolved=True, source_tag="00int")
    sub_est2020int = _make_place_vintage(
        list(range(2010, 2021)),
        include_dissolved=True,
        source_tag="2020int",
    )
    sub_est2024 = _make_place_vintage(list(range(2020, 2025)), include_dissolved=False, source_tag="2024")

    history = assemble_place_population_history(sub_est00int, sub_est2020int, sub_est2024)
    history = attach_crosswalk(history, _crosswalk_fixture())

    expected_cols = {
        "place_fips",
        "place_name",
        "county_fips",
        "year",
        "population",
        "vintage_source",
        "historical_only",
    }
    assert expected_cols.issubset(history.columns)

    history["historical_only"] = history["historical_only"].fillna(False).astype(bool)
    validate_place_history(history)

    # 2 active places x 25 years + 1 dissolved place x 20 years = 70 rows
    assert len(history) == 70
    assert set(history["year"].unique()) == set(range(2000, 2025))

    # Handoff windows
    assert set(history[history["year"] <= 2009]["vintage_source"]) == {"sub-est00int"}
    assert set(history[(history["year"] >= 2010) & (history["year"] <= 2019)]["vintage_source"]) == {
        "sub-est2020int",
    }
    assert set(history[history["year"] >= 2020]["vintage_source"]) == {"sub-est2024"}


def test_dissolved_place_present_through_2019_only() -> None:
    """Dissolved place rows stop after 2019 and are marked historical-only."""
    sub_est00int = _make_place_vintage(list(range(2000, 2011)), include_dissolved=True, source_tag="00int")
    sub_est2020int = _make_place_vintage(
        list(range(2010, 2021)),
        include_dissolved=True,
        source_tag="2020int",
    )
    sub_est2024 = _make_place_vintage(list(range(2020, 2025)), include_dissolved=False, source_tag="2024")

    history = attach_crosswalk(
        assemble_place_population_history(sub_est00int, sub_est2020int, sub_est2024),
        _crosswalk_fixture(),
    )
    history["historical_only"] = history["historical_only"].fillna(False).astype(bool)

    bantry = history[history["place_fips"] == "3804740"]
    assert set(bantry["year"]) == set(range(2000, 2020))
    assert bantry["historical_only"].all()

    active_places = history[~history["historical_only"]]["place_fips"].nunique()
    assert active_places == 2


def test_fargo_2024_population_comes_from_sub_est2024_window() -> None:
    """Spot-check that 2024 values come from the postcensal source."""
    sub_est00int = _make_place_vintage(list(range(2000, 2011)), include_dissolved=True, source_tag="00int")
    sub_est2020int = _make_place_vintage(
        list(range(2010, 2021)),
        include_dissolved=True,
        source_tag="2020int",
    )
    sub_est2024 = _make_place_vintage(list(range(2020, 2025)), include_dissolved=False, source_tag="2024")

    history = attach_crosswalk(
        assemble_place_population_history(sub_est00int, sub_est2020int, sub_est2024),
        _crosswalk_fixture(),
    )
    fargo_2024 = history[(history["place_fips"] == "3825700") & (history["year"] == 2024)].iloc[0]

    # base 110000 with offset 4 for 2024 in the synthetic 2020-2024 window
    assert fargo_2024["population"] == 110_004
    assert fargo_2024["vintage_source"] == "sub-est2024"


def test_tier_builder_uses_active_2024_rows_only() -> None:
    """Tier assignment excludes historical-only places and uses year 2024."""
    history = pd.DataFrame(
        [
            {
                "place_fips": "3825700",
                "year": 2024,
                "population": 130_000,
                "historical_only": False,
            },
            {
                "place_fips": "3807200",
                "year": 2024,
                "population": 74_000,
                "historical_only": False,
            },
            {
                "place_fips": "3804740",
                "year": 2019,
                "population": 12,
                "historical_only": True,
            },
            {
                "place_fips": "3804740",
                "year": 2024,
                "population": 10,
                "historical_only": True,
            },
        ],
    )

    tiers = build_tiers_from_history(history)
    assert set(tiers["place_fips"]) == {"3825700", "3807200"}
    assert set(tiers["confidence_tier"]) == {"HIGH"}
