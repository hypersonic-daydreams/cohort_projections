"""
Tests for PP-005 WS-B multi-county place allocation (ADR-058).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cohort_projections.data.process.multicounty_allocation import (
    get_multicounty_config,
    identify_multicounty_places,
    load_allocation_weights,
    prepare_multicounty_share_history,
    reaggregate_multicounty_place,
    split_multicounty_place,
    split_multicounty_shares,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_crosswalk() -> pd.DataFrame:
    """Primary crosswalk with a mix of single and multi-county places."""
    return pd.DataFrame(
        {
            "place_fips": [
                "3800100",
                "3824260",
                "3832300",
                "3845740",
            ],
            "county_fips": [
                "38077",
                "38073",
                "38017",
                "38051",
            ],
            "assignment_type": [
                "single_county",
                "multi_county_primary",
                "multi_county_primary",
                "multi_county_primary",
            ],
            "place_name": [
                "Abercrombie city",
                "Enderlin city",
                "Grandin city",
                "Lehr city",
            ],
            "confidence_tier": [
                "EXCLUDED",
                "LOWER",
                "EXCLUDED",
                "EXCLUDED",
            ],
        }
    )


@pytest.fixture()
def sample_weights() -> dict[str, dict[str, float]]:
    """Allocation weights for two multicounty places."""
    return {
        "3824260": {"38073": 0.9895, "38017": 0.0105},
        "3832300": {"38017": 0.8629, "38097": 0.1371},
        "3845740": {"38051": 0.6013, "38047": 0.3987},
    }


@pytest.fixture()
def sample_share_history() -> pd.DataFrame:
    """Synthetic share history for multicounty places in their primary counties."""
    rows = []
    for year in range(2000, 2005):
        # Enderlin in primary county 38073
        rows.append(
            {
                "county_fips": "38073",
                "year": year,
                "place_fips": "3824260",
                "share_raw": 0.20 + 0.001 * (year - 2000),
                "row_type": "place",
            }
        )
        # Grandin in primary county 38017
        rows.append(
            {
                "county_fips": "38017",
                "year": year,
                "place_fips": "3832300",
                "share_raw": 0.05 + 0.001 * (year - 2000),
                "row_type": "place",
            }
        )
        # Single-county place in 38077
        rows.append(
            {
                "county_fips": "38077",
                "year": year,
                "place_fips": "3800100",
                "share_raw": 0.10,
                "row_type": "place",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def multicounty_detail_csv(tmp_path: Path) -> Path:
    """Write a temporary multicounty detail CSV."""
    detail = pd.DataFrame(
        {
            "state_fips": ["38", "38", "38", "38", "38", "38"],
            "place_fips": [
                "3824260",
                "3824260",
                "3832300",
                "3832300",
                "3845740",
                "3845740",
            ],
            "place_name": [
                "Enderlin city",
                "Enderlin city",
                "Grandin city",
                "Grandin city",
                "Lehr city",
                "Lehr city",
            ],
            "county_fips": [
                "38073",
                "38017",
                "38017",
                "38097",
                "38051",
                "38047",
            ],
            "assignment_type": ["multi_county_primary"] * 6,
            "area_share": [0.9895, 0.0105, 0.8629, 0.1371, 0.6013, 0.3987],
            "county_rank": [1, 2, 1, 2, 1, 2],
            "is_primary": [True, False, True, False, True, False],
            "historical_only": [False] * 6,
            "source_vintage": ["2020"] * 6,
            "source_method": ["tiger_overlay"] * 6,
        }
    )
    csv_path = tmp_path / "multicounty_detail.csv"
    detail.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def crosswalk_csv(tmp_path: Path, sample_crosswalk: pd.DataFrame) -> Path:
    """Write a temporary primary crosswalk CSV."""
    csv_path = tmp_path / "crosswalk.csv"
    sample_crosswalk.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Tests: identify_multicounty_places
# ---------------------------------------------------------------------------


class TestIdentifyMulticountyPlaces:
    """Tests for identify_multicounty_places."""

    def test_returns_multicounty_place_fips(
        self, sample_crosswalk: pd.DataFrame
    ) -> None:
        """Returns only places with multi_county_primary assignment type."""
        result = identify_multicounty_places(sample_crosswalk)
        assert result == ["3824260", "3832300", "3845740"]

    def test_returns_empty_for_all_single_county(self) -> None:
        """Returns empty list when all places are single-county."""
        xw = pd.DataFrame(
            {
                "place_fips": ["3800100", "3800200"],
                "assignment_type": ["single_county", "single_county"],
            }
        )
        assert identify_multicounty_places(xw) == []

    def test_returns_empty_for_empty_crosswalk(self) -> None:
        """Returns empty list for empty input DataFrame."""
        xw = pd.DataFrame(
            columns=["place_fips", "assignment_type"]
        )
        assert identify_multicounty_places(xw) == []

    def test_raises_on_missing_columns(self) -> None:
        """Raises ValueError when required columns are missing."""
        xw = pd.DataFrame({"place_fips": ["3800100"]})
        with pytest.raises(ValueError, match="missing required columns"):
            identify_multicounty_places(xw)


# ---------------------------------------------------------------------------
# Tests: load_allocation_weights
# ---------------------------------------------------------------------------


class TestLoadAllocationWeights:
    """Tests for load_allocation_weights."""

    def test_loads_correct_weights(
        self, crosswalk_csv: Path, multicounty_detail_csv: Path
    ) -> None:
        """Weights are loaded and normalized correctly."""
        weights = load_allocation_weights(crosswalk_csv, multicounty_detail_csv)
        assert len(weights) == 3
        assert "3824260" in weights
        assert "3832300" in weights
        assert "3845740" in weights

        # Weights sum to 1 within each place.
        for place_fips, county_weights in weights.items():
            np.testing.assert_allclose(
                sum(county_weights.values()),
                1.0,
                atol=1e-9,
                err_msg=f"Weights for {place_fips} do not sum to 1.",
            )

    def test_weights_have_correct_counties(
        self, crosswalk_csv: Path, multicounty_detail_csv: Path
    ) -> None:
        """Each place has the expected constituent counties."""
        weights = load_allocation_weights(crosswalk_csv, multicounty_detail_csv)
        assert set(weights["3824260"].keys()) == {"38073", "38017"}
        assert set(weights["3832300"].keys()) == {"38017", "38097"}
        assert set(weights["3845740"].keys()) == {"38051", "38047"}

    def test_raises_on_missing_crosswalk(
        self, tmp_path: Path, multicounty_detail_csv: Path
    ) -> None:
        """Raises FileNotFoundError when crosswalk path is invalid."""
        bad_path = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            load_allocation_weights(bad_path, multicounty_detail_csv)

    def test_raises_on_missing_detail(
        self, crosswalk_csv: Path, tmp_path: Path
    ) -> None:
        """Raises FileNotFoundError when detail path is invalid."""
        bad_path = tmp_path / "nonexistent.csv"
        with pytest.raises(FileNotFoundError):
            load_allocation_weights(crosswalk_csv, bad_path)

    def test_raises_on_missing_columns(
        self, crosswalk_csv: Path, tmp_path: Path
    ) -> None:
        """Raises ValueError when required columns are absent in detail CSV."""
        bad_csv = tmp_path / "bad_detail.csv"
        pd.DataFrame({"foo": [1]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_allocation_weights(crosswalk_csv, bad_csv)


# ---------------------------------------------------------------------------
# Tests: split_multicounty_place
# ---------------------------------------------------------------------------


class TestSplitMulticountyPlace:
    """Tests for split_multicounty_place."""

    def test_split_preserves_total(
        self, sample_weights: dict[str, dict[str, float]]
    ) -> None:
        """Sum of allocated populations equals input population."""
        result = split_multicounty_place("3824260", 1000.0, sample_weights)
        np.testing.assert_allclose(
            sum(result.values()), 1000.0, atol=1e-9
        )

    def test_split_two_county_proportions(
        self, sample_weights: dict[str, dict[str, float]]
    ) -> None:
        """Allocations match expected proportions for 2-county place."""
        result = split_multicounty_place("3824260", 1000.0, sample_weights)
        assert set(result.keys()) == {"38073", "38017"}
        np.testing.assert_allclose(
            result["38073"], 1000.0 * 0.9895, atol=1.0
        )

    def test_split_three_county_proportions(self) -> None:
        """Split works correctly for 3-county place."""
        weights = {
            "3899999": {
                "38001": 0.5,
                "38002": 0.3,
                "38003": 0.2,
            }
        }
        result = split_multicounty_place("3899999", 500.0, weights)
        assert len(result) == 3
        np.testing.assert_allclose(sum(result.values()), 500.0, atol=1e-9)
        np.testing.assert_allclose(result["38001"], 250.0, atol=1e-9)
        np.testing.assert_allclose(result["38002"], 150.0, atol=1e-9)
        np.testing.assert_allclose(result["38003"], 100.0, atol=1e-9)

    def test_split_zero_population(
        self, sample_weights: dict[str, dict[str, float]]
    ) -> None:
        """Zero population allocates zero to all counties."""
        result = split_multicounty_place("3824260", 0.0, sample_weights)
        assert all(v == 0.0 for v in result.values())

    def test_split_negative_population_raises(
        self, sample_weights: dict[str, dict[str, float]]
    ) -> None:
        """Raises ValueError for negative population."""
        with pytest.raises(ValueError, match="non-negative"):
            split_multicounty_place("3824260", -100.0, sample_weights)

    def test_split_unknown_place_raises(
        self, sample_weights: dict[str, dict[str, float]]
    ) -> None:
        """Raises ValueError for unknown place FIPS."""
        with pytest.raises(ValueError, match="not found"):
            split_multicounty_place("9999999", 100.0, sample_weights)

    def test_roundtrip_split_sum_invariant(self) -> None:
        """Split/sum roundtrip always preserves total population."""
        weights = {
            "3812345": {
                "38010": 0.75,
                "38020": 0.25,
            }
        }
        for pop in [0.0, 1.0, 100.0, 1e6, 0.001]:
            result = split_multicounty_place("3812345", pop, weights)
            np.testing.assert_allclose(
                sum(result.values()),
                pop,
                atol=1e-9,
                err_msg=f"Roundtrip failed for population={pop}",
            )


# ---------------------------------------------------------------------------
# Tests: split_multicounty_shares
# ---------------------------------------------------------------------------


class TestSplitMulticountyShares:
    """Tests for split_multicounty_shares."""

    def test_creates_synthetic_rows_for_nonprimary_counties(
        self,
        sample_share_history: pd.DataFrame,
        sample_weights: dict[str, dict[str, float]],
    ) -> None:
        """Creates share rows for the non-primary county."""
        result = split_multicounty_shares(
            share_history=sample_share_history,
            weights=sample_weights,
            place_fips="3824260",
            primary_county_fips="38073",
        )
        assert not result.empty
        assert (result["county_fips"] == "38017").all()
        assert (result["place_fips"] == "3824260").all()
        assert len(result) == 5  # 5 years of data

    def test_synthetic_shares_are_scaled_correctly(
        self,
        sample_share_history: pd.DataFrame,
        sample_weights: dict[str, dict[str, float]],
    ) -> None:
        """Synthetic shares reflect weight ratio."""
        result = split_multicounty_shares(
            share_history=sample_share_history,
            weights=sample_weights,
            place_fips="3824260",
            primary_county_fips="38073",
        )
        primary_weight = sample_weights["3824260"]["38073"]
        secondary_weight = sample_weights["3824260"]["38017"]
        ratio = secondary_weight / primary_weight

        # For year 2000, primary share is 0.20
        year_2000 = result[result["year"] == 2000]
        expected_share = 0.20 * ratio
        np.testing.assert_allclose(
            float(year_2000["share_raw"].iloc[0]),
            expected_share,
            rtol=1e-6,
        )

    def test_does_not_include_primary_county_rows(
        self,
        sample_share_history: pd.DataFrame,
        sample_weights: dict[str, dict[str, float]],
    ) -> None:
        """Output does not include primary county rows."""
        result = split_multicounty_shares(
            share_history=sample_share_history,
            weights=sample_weights,
            place_fips="3824260",
            primary_county_fips="38073",
        )
        assert "38073" not in result["county_fips"].values

    def test_empty_result_for_unknown_place(
        self,
        sample_share_history: pd.DataFrame,
        sample_weights: dict[str, dict[str, float]],
    ) -> None:
        """Returns empty DataFrame for place with no history rows."""
        result = split_multicounty_shares(
            share_history=sample_share_history,
            weights=sample_weights,
            place_fips="3845740",
            primary_county_fips="38051",
        )
        assert result.empty

    def test_raises_for_unknown_place_in_weights(
        self,
        sample_share_history: pd.DataFrame,
        sample_weights: dict[str, dict[str, float]],
    ) -> None:
        """Raises ValueError when place not in weights."""
        with pytest.raises(ValueError, match="not found"):
            split_multicounty_shares(
                share_history=sample_share_history,
                weights=sample_weights,
                place_fips="9999999",
                primary_county_fips="38073",
            )

    def test_raises_for_unknown_primary_county(
        self,
        sample_share_history: pd.DataFrame,
        sample_weights: dict[str, dict[str, float]],
    ) -> None:
        """Raises ValueError when primary county not in weights."""
        with pytest.raises(ValueError, match="not in weights"):
            split_multicounty_shares(
                share_history=sample_share_history,
                weights=sample_weights,
                place_fips="3824260",
                primary_county_fips="99999",
            )


# ---------------------------------------------------------------------------
# Tests: reaggregate_multicounty_place
# ---------------------------------------------------------------------------


class TestReaggregateMulticountyPlace:
    """Tests for reaggregate_multicounty_place."""

    def test_sums_county_portions_correctly(
        self, sample_weights: dict[str, dict[str, float]]
    ) -> None:
        """Reaggregated total equals sum of county portions."""
        proj_38073 = pd.DataFrame(
            {
                "year": [2025, 2030],
                "place_fips": ["3824260", "3824260"],
                "projected_population": [800.0, 850.0],
            }
        )
        proj_38017 = pd.DataFrame(
            {
                "year": [2025, 2030],
                "place_fips": ["3824260", "3824260"],
                "projected_population": [10.0, 12.0],
            }
        )
        county_projections = {
            "38073": proj_38073,
            "38017": proj_38017,
        }
        result = reaggregate_multicounty_place(
            county_projections=county_projections,
            place_fips="3824260",
            weights=sample_weights,
        )
        assert len(result) == 2
        np.testing.assert_allclose(
            result[result["year"] == 2025]["place_total"].values[0],
            810.0,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            result[result["year"] == 2030]["place_total"].values[0],
            862.0,
            atol=1e-9,
        )

    def test_raises_for_unknown_place(
        self, sample_weights: dict[str, dict[str, float]]
    ) -> None:
        """Raises ValueError for place not in weights."""
        with pytest.raises(ValueError, match="not found"):
            reaggregate_multicounty_place(
                county_projections={},
                place_fips="9999999",
                weights=sample_weights,
            )

    def test_raises_when_no_projection_data(
        self, sample_weights: dict[str, dict[str, float]]
    ) -> None:
        """Raises ValueError when no county has projection data for the place."""
        empty_proj = pd.DataFrame(
            columns=["year", "place_fips", "projected_population"]
        )
        with pytest.raises(ValueError, match="No projection data"):
            reaggregate_multicounty_place(
                county_projections={"38073": empty_proj},
                place_fips="3824260",
                weights=sample_weights,
            )

    def test_handles_partial_county_data(
        self, sample_weights: dict[str, dict[str, float]]
    ) -> None:
        """Reaggregates correctly when only primary county has data."""
        proj_38073 = pd.DataFrame(
            {
                "year": [2025],
                "place_fips": ["3824260"],
                "projected_population": [900.0],
            }
        )
        county_projections = {"38073": proj_38073}
        result = reaggregate_multicounty_place(
            county_projections=county_projections,
            place_fips="3824260",
            weights=sample_weights,
        )
        assert len(result) == 1
        np.testing.assert_allclose(
            result["place_total"].values[0], 900.0, atol=1e-9
        )


# ---------------------------------------------------------------------------
# Tests: get_multicounty_config
# ---------------------------------------------------------------------------


class TestGetMulticountyConfig:
    """Tests for get_multicounty_config."""

    def test_returns_defaults_when_absent(self) -> None:
        """Returns sensible defaults when config section is absent."""
        cfg = get_multicounty_config({})
        assert cfg["enabled"] is False
        assert cfg["allocation_method"] == "area_share"
        assert "multicounty_detail" in cfg["multicounty_detail_path"]

    def test_reads_enabled_flag(self) -> None:
        """Reads enabled flag from config."""
        cfg = get_multicounty_config(
            {
                "place_projections": {
                    "multicounty_allocation": {"enabled": True}
                }
            }
        )
        assert cfg["enabled"] is True

    def test_reads_allocation_method(self) -> None:
        """Reads allocation_method from config."""
        cfg = get_multicounty_config(
            {
                "place_projections": {
                    "multicounty_allocation": {
                        "allocation_method": "population_share"
                    }
                }
            }
        )
        assert cfg["allocation_method"] == "population_share"


# ---------------------------------------------------------------------------
# Tests: prepare_multicounty_share_history
# ---------------------------------------------------------------------------


class TestPrepareMulticountyShareHistory:
    """Tests for prepare_multicounty_share_history."""

    def test_augments_history_with_synthetic_rows(
        self,
        sample_share_history: pd.DataFrame,
        sample_crosswalk: pd.DataFrame,
        sample_weights: dict[str, dict[str, float]],
    ) -> None:
        """Augmented history has more rows than original."""
        augmented = prepare_multicounty_share_history(
            share_history=sample_share_history,
            crosswalk=sample_crosswalk,
            weights=sample_weights,
            multicounty_place_fips=["3824260", "3832300"],
        )
        assert len(augmented) > len(sample_share_history)

    def test_original_rows_preserved(
        self,
        sample_share_history: pd.DataFrame,
        sample_crosswalk: pd.DataFrame,
        sample_weights: dict[str, dict[str, float]],
    ) -> None:
        """Original share history rows are not modified."""
        augmented = prepare_multicounty_share_history(
            share_history=sample_share_history,
            crosswalk=sample_crosswalk,
            weights=sample_weights,
            multicounty_place_fips=["3824260"],
        )
        # The original rows for 3824260 in county 38073 should still be present.
        original_rows = augmented[
            (augmented["place_fips"] == "3824260")
            & (augmented["county_fips"] == "38073")
        ]
        assert len(original_rows) == 5

    def test_returns_unchanged_when_no_multicounty(
        self,
        sample_share_history: pd.DataFrame,
        sample_crosswalk: pd.DataFrame,
        sample_weights: dict[str, dict[str, float]],
    ) -> None:
        """Returns unchanged history when no multicounty places given."""
        result = prepare_multicounty_share_history(
            share_history=sample_share_history,
            crosswalk=sample_crosswalk,
            weights=sample_weights,
            multicounty_place_fips=[],
        )
        assert len(result) == len(sample_share_history)

    def test_synthetic_rows_have_correct_county(
        self,
        sample_share_history: pd.DataFrame,
        sample_crosswalk: pd.DataFrame,
        sample_weights: dict[str, dict[str, float]],
    ) -> None:
        """Synthetic rows for Grandin (3832300) appear under county 38097."""
        augmented = prepare_multicounty_share_history(
            share_history=sample_share_history,
            crosswalk=sample_crosswalk,
            weights=sample_weights,
            multicounty_place_fips=["3832300"],
        )
        synthetic = augmented[
            (augmented["place_fips"] == "3832300")
            & (augmented["county_fips"] == "38097")
        ]
        assert len(synthetic) == 5  # 5 years


# ---------------------------------------------------------------------------
# Tests: edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and integration tests."""

    def test_place_entirely_in_one_county(self) -> None:
        """A place with only one county in weights acts as passthrough."""
        weights = {"3800100": {"38077": 1.0}}
        result = split_multicounty_place("3800100", 500.0, weights)
        assert len(result) == 1
        np.testing.assert_allclose(result["38077"], 500.0, atol=1e-9)

    def test_very_small_weight(self) -> None:
        """Very small allocation weights produce correspondingly small shares."""
        weights = {"3899999": {"38001": 0.999, "38002": 0.001}}
        result = split_multicounty_place("3899999", 10000.0, weights)
        np.testing.assert_allclose(result["38002"], 10.0, atol=1e-6)
        np.testing.assert_allclose(
            sum(result.values()), 10000.0, atol=1e-9
        )

    def test_equal_split(self) -> None:
        """Equal weights produce equal allocations."""
        weights = {"3899999": {"38001": 0.5, "38002": 0.5}}
        result = split_multicounty_place("3899999", 1000.0, weights)
        np.testing.assert_allclose(result["38001"], 500.0, atol=1e-9)
        np.testing.assert_allclose(result["38002"], 500.0, atol=1e-9)

    def test_identify_returns_sorted_list(self) -> None:
        """Result is sorted alphabetically."""
        xw = pd.DataFrame(
            {
                "place_fips": ["3899999", "3811111", "3855555"],
                "assignment_type": [
                    "multi_county_primary",
                    "multi_county_primary",
                    "single_county",
                ],
            }
        )
        result = identify_multicounty_places(xw)
        assert result == ["3811111", "3899999"]

    def test_split_reaggregate_roundtrip(self) -> None:
        """Split then reaggregate recovers original total population."""
        weights = {
            "3812345": {
                "38010": 0.6,
                "38020": 0.3,
                "38030": 0.1,
            }
        }
        total_pop = 5000.0
        allocations = split_multicounty_place("3812345", total_pop, weights)

        # Build county projections from the allocations.
        county_projections = {}
        for cfips, pop in allocations.items():
            county_projections[cfips] = pd.DataFrame(
                {
                    "year": [2025],
                    "place_fips": ["3812345"],
                    "projected_population": [pop],
                }
            )

        reagg = reaggregate_multicounty_place(
            county_projections=county_projections,
            place_fips="3812345",
            weights=weights,
        )
        np.testing.assert_allclose(
            reagg["place_total"].values[0],
            total_pop,
            atol=1e-9,
        )
