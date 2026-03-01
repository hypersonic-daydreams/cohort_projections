"""
Tests for place->county crosswalk assembly and tier assignment (PP-003 Phase 1).
"""

from __future__ import annotations

import pandas as pd

from scripts.data.build_place_county_crosswalk import (
    add_tiers_to_crosswalk,
    assign_confidence_tiers,
    build_place_county_crosswalk,
    validate_crosswalk,
)


def _active_places_fixture() -> pd.DataFrame:
    """Create a small active-place universe with one multi-county example."""
    return pd.DataFrame(
        [
            {"state_fips": "38", "place_fips": "3825700", "place_name": "Fargo city"},
            {"state_fips": "38", "place_fips": "3807200", "place_name": "Bismarck city"},
            {"state_fips": "38", "place_fips": "3832060", "place_name": "Grand Forks city"},
            {"state_fips": "38", "place_fips": "3853380", "place_name": "Minot city"},
            {"state_fips": "38", "place_fips": "3884780", "place_name": "West Fargo city"},
            {"state_fips": "38", "place_fips": "3886220", "place_name": "Williston city"},
            {"state_fips": "38", "place_fips": "3899999", "place_name": "Splitville city"},
        ],
    )


def _overlaps_fixture() -> pd.DataFrame:
    """Create synthetic overlaps with one multi-county place."""
    return pd.DataFrame(
        [
            {"place_fips": "3825700", "county_fips": "38017", "area_share": 1.0},
            {"place_fips": "3807200", "county_fips": "38015", "area_share": 1.0},
            {"place_fips": "3832060", "county_fips": "38035", "area_share": 1.0},
            {"place_fips": "3853380", "county_fips": "38101", "area_share": 1.0},
            {"place_fips": "3884780", "county_fips": "38017", "area_share": 1.0},
            {"place_fips": "3886220", "county_fips": "38105", "area_share": 1.0},
            # Multi-county example (primary should be 38017)
            {"place_fips": "3899999", "county_fips": "38017", "area_share": 0.70},
            {"place_fips": "3899999", "county_fips": "38073", "area_share": 0.30},
        ],
    )


def test_primary_crosswalk_has_one_row_per_active_place_plus_two_historical() -> None:
    """All active places map once; dissolved historical rows are appended."""
    active_places = _active_places_fixture()
    overlaps = _overlaps_fixture()

    primary, _detail = build_place_county_crosswalk(overlaps=overlaps, active_places=active_places)

    non_historical = primary[~primary["historical_only"]]
    assert len(non_historical) == len(active_places)

    historical = primary[primary["historical_only"]]
    assert set(historical["place_fips"]) == {"3804740", "3814140"}
    assert len(primary) == len(active_places) + 2


def test_crosswalk_validates_assignment_types_fips_and_area_shares() -> None:
    """Crosswalk passes PP3-S03 schema invariants."""
    active_places = _active_places_fixture()
    overlaps = _overlaps_fixture()
    primary, _detail = build_place_county_crosswalk(overlaps=overlaps, active_places=active_places)

    validate_crosswalk(primary, expected_active_places=len(active_places))
    assert primary["assignment_type"].isin({"single_county", "multi_county_primary"}).all()
    assert primary["place_fips"].str.fullmatch(r"\d{7}").all()
    assert primary["county_fips"].str.fullmatch(r"\d{5}").all()
    assert ((primary["area_share"] > 0) & (primary["area_share"] <= 1.0)).all()


def test_multi_county_place_appears_in_primary_and_detail() -> None:
    """A multi-county place gets a primary assignment and full detail rows."""
    active_places = _active_places_fixture()
    overlaps = _overlaps_fixture()
    primary, detail = build_place_county_crosswalk(overlaps=overlaps, active_places=active_places)

    split_primary = primary[primary["place_fips"] == "3899999"].iloc[0]
    assert split_primary["assignment_type"] == "multi_county_primary"
    assert split_primary["county_fips"] == "38017"
    assert split_primary["area_share"] == 0.70

    split_detail = detail[detail["place_fips"] == "3899999"]
    assert len(split_detail) == 2
    assert set(split_detail["county_fips"]) == {"38017", "38073"}
    assert split_detail["is_primary"].sum() == 1


def test_major_place_spot_checks_match_expected_counties() -> None:
    """Major ND places map to expected county FIPS assignments."""
    active_places = _active_places_fixture()
    overlaps = _overlaps_fixture()
    primary, _detail = build_place_county_crosswalk(overlaps=overlaps, active_places=active_places)

    checks = {
        "Fargo city": "38017",
        "Bismarck city": "38015",
        "Grand Forks city": "38035",
        "Minot city": "38101",
        "West Fargo city": "38017",
        "Williston city": "38105",
    }
    for place_name, expected_county in checks.items():
        row = primary[primary["place_name"] == place_name]
        assert len(row) == 1
        assert row.iloc[0]["county_fips"] == expected_county


def test_tier_assignment_matches_phase1_counts_and_threshold_rules() -> None:
    """Tier assignment returns 9/9/72/265 and handles threshold edges correctly."""
    rows: list[dict[str, object]] = []

    for i in range(9):
        rows.append({"place_fips": f"38{(10000 + i):05d}", "population_2024": 11_000 + i})
    for i in range(9):
        rows.append({"place_fips": f"38{(20000 + i):05d}", "population_2024": 5_000 + i})
    for i in range(72):
        rows.append({"place_fips": f"38{(30000 + i):05d}", "population_2024": 1_200 + i})
    for i in range(265):
        rows.append({"place_fips": f"38{(40000 + i):05d}", "population_2024": 300 + i % 150})

    tiers = assign_confidence_tiers(pd.DataFrame(rows))
    counts = tiers["confidence_tier"].value_counts().to_dict()
    assert counts == {
        "EXCLUDED": 265,
        "LOWER": 72,
        "HIGH": 9,
        "MODERATE": 9,
    }

    threshold_probe = pd.DataFrame(
        [
            {"place_fips": "3800001", "population_2024": 10_000},
            {"place_fips": "3800002", "population_2024": 10_001},
            {"place_fips": "3800003", "population_2024": 2_500},
            {"place_fips": "3800004", "population_2024": 500},
            {"place_fips": "3800005", "population_2024": 499},
        ],
    )
    probe_tiers = assign_confidence_tiers(threshold_probe).set_index("place_fips")
    assert probe_tiers.loc["3800001", "confidence_tier"] == "MODERATE"
    assert probe_tiers.loc["3800002", "confidence_tier"] == "HIGH"
    assert probe_tiers.loc["3800003", "confidence_tier"] == "MODERATE"
    assert probe_tiers.loc["3800004", "confidence_tier"] == "LOWER"
    assert probe_tiers.loc["3800005", "confidence_tier"] == "EXCLUDED"


def test_add_tiers_to_crosswalk_marks_historical_rows_as_excluded() -> None:
    """Historical-only rows are forced to EXCLUDED and non-boundary."""
    crosswalk = pd.DataFrame(
        [
            {
                "place_fips": "3825700",
                "county_fips": "38017",
                "historical_only": False,
            },
            {
                "place_fips": "3804740",
                "county_fips": "38049",
                "historical_only": True,
            },
        ],
    )
    tiers = pd.DataFrame(
        [
            {"place_fips": "3825700", "confidence_tier": "HIGH", "tier_boundary": False},
            {"place_fips": "3804740", "confidence_tier": "LOWER", "tier_boundary": True},
        ],
    )

    enriched = add_tiers_to_crosswalk(crosswalk, tiers).set_index("place_fips")
    assert enriched.loc["3825700", "confidence_tier"] == "HIGH"
    assert bool(enriched.loc["3825700", "tier_boundary"]) is False

    assert enriched.loc["3804740", "confidence_tier"] == "EXCLUDED"
    assert bool(enriched.loc["3804740", "tier_boundary"]) is False
