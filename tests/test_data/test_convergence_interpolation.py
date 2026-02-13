"""
Tests for age-specific convergence interpolation (Phase 2).

Validates the 5-10-5 convergence schedule applied to Phase 1 residual
migration rates.  Uses synthetic data for unit tests and real Phase 1
output for integration tests.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cohort_projections.data.process.convergence_interpolation import (
    _map_config_window_to_periods,
    calculate_age_specific_convergence,
    compute_period_window_averages,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 18 five-year age groups used in Phase 1
AGE_GROUPS = [
    "0-4",
    "5-9",
    "10-14",
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80-84",
    "85+",
]
SEX_CATEGORIES = ["Male", "Female"]

# 36 cells per county per year = 18 age groups x 2 sexes
CELLS_PER_COUNTY = len(AGE_GROUPS) * len(SEX_CATEGORIES)  # 36

# All 53 ND counties (FIPS codes from 38001 to 38105, odd numbers)
ND_COUNTY_FIPS = [
    "38001",
    "38003",
    "38005",
    "38007",
    "38009",
    "38011",
    "38013",
    "38015",
    "38017",
    "38019",
    "38021",
    "38023",
    "38025",
    "38027",
    "38029",
    "38031",
    "38033",
    "38035",
    "38037",
    "38039",
    "38041",
    "38043",
    "38045",
    "38047",
    "38049",
    "38051",
    "38053",
    "38055",
    "38057",
    "38059",
    "38061",
    "38063",
    "38065",
    "38067",
    "38069",
    "38071",
    "38073",
    "38075",
    "38077",
    "38079",
    "38081",
    "38083",
    "38085",
    "38087",
    "38089",
    "38091",
    "38093",
    "38095",
    "38097",
    "38099",
    "38101",
    "38103",
    "38105",
]

# The 5 Phase 1 periods
PERIODS = [
    (2000, 2005),
    (2005, 2010),
    (2010, 2015),
    (2015, 2020),
    (2020, 2024),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_period_rates(
    counties: list[str],
    periods: list[tuple[int, int]] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic Phase 1 period rates.

    Args:
        counties: List of county FIPS codes.
        periods: List of (start, end) tuples.  Defaults to all 5 periods.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame matching the Phase 1 output schema.
    """
    if periods is None:
        periods = PERIODS

    np.random.seed(seed)
    records = []
    for county in counties:
        for ps, pe in periods:
            for ag in AGE_GROUPS:
                for sex in SEX_CATEGORIES:
                    # Deterministic rate based on hash of identifiers + noise
                    base = hash((county, ag, sex)) % 100 / 1000.0 - 0.05
                    # Add period variation
                    period_shift = (ps - 2000) * 0.002
                    rate = base + period_shift + np.random.normal(0, 0.01)
                    records.append(
                        {
                            "county_fips": county,
                            "age_group": ag,
                            "sex": sex,
                            "period_start": ps,
                            "period_end": pe,
                            "pop_start": 1000.0,
                            "expected_pop": 980.0,
                            "pop_end": 1010.0,
                            "net_migration": rate * 1000,
                            "migration_rate": rate,
                        }
                    )
    return pd.DataFrame(records)


@pytest.fixture
def three_county_rates() -> pd.DataFrame:
    """Synthetic period rates for 3 counties, all 5 periods."""
    return _make_period_rates(["38001", "38017", "38105"])


@pytest.fixture
def known_rates() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Window averages with known exact values for convergence testing.

    Returns (recent, medium, longterm) where:
      - recent  rate = 0.10 for all cells
      - medium  rate = 0.05 for all cells
      - longterm rate = 0.02 for all cells
    """
    counties = ["38001", "38017"]
    records = [
        {"county_fips": county, "age_group": ag, "sex": sex}
        for county in counties
        for ag in AGE_GROUPS
        for sex in SEX_CATEGORIES
    ]

    base = pd.DataFrame(records)
    recent = base.copy()
    recent["migration_rate"] = 0.10
    medium = base.copy()
    medium["migration_rate"] = 0.05
    longterm = base.copy()
    longterm["migration_rate"] = 0.02

    return recent, medium, longterm


@pytest.fixture
def mixed_sign_rates() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Window averages with a mix of positive and negative rates.

    Cell 1 (38001, 0-4, Male):  recent=+0.08, medium=+0.04, longterm=+0.01
    Cell 2 (38001, 0-4, Female): recent=-0.06, medium=-0.03, longterm=-0.01
    """
    cells = [
        {"county_fips": "38001", "age_group": "0-4", "sex": "Male"},
        {"county_fips": "38001", "age_group": "0-4", "sex": "Female"},
    ]
    recent = pd.DataFrame(cells)
    recent["migration_rate"] = [0.08, -0.06]
    medium = pd.DataFrame(cells)
    medium["migration_rate"] = [0.04, -0.03]
    longterm = pd.DataFrame(cells)
    longterm["migration_rate"] = [0.01, -0.01]

    return recent, medium, longterm


# ---------------------------------------------------------------------------
# Tests: Period-to-window mapping
# ---------------------------------------------------------------------------


class TestPeriodWindowMapping:
    """Tests for _map_config_window_to_periods."""

    def test_recent_window_maps_to_single_period(self) -> None:
        """recent_period [2022, 2024] maps only to (2020, 2024)."""
        result = _map_config_window_to_periods([2022, 2024], PERIODS)
        assert result == [(2020, 2024)]

    def test_medium_window_maps_to_three_periods(self) -> None:
        """medium_period [2014, 2024] maps to 3 periods."""
        result = _map_config_window_to_periods([2014, 2024], PERIODS)
        assert result == [(2010, 2015), (2015, 2020), (2020, 2024)]

    def test_longterm_window_maps_to_all_periods(self) -> None:
        """longterm_period [2000, 2024] maps to all 5 periods."""
        result = _map_config_window_to_periods([2000, 2024], PERIODS)
        assert result == list(PERIODS)


# ---------------------------------------------------------------------------
# Tests: Window averaging
# ---------------------------------------------------------------------------


class TestWindowAveraging:
    """Tests for compute_period_window_averages."""

    def test_recent_averages_only_recent_period(self, three_county_rates: pd.DataFrame) -> None:
        """Recent average uses only the (2020,2024) period."""
        recent, _, _ = compute_period_window_averages(
            three_county_rates,
            recent_periods=[(2020, 2024)],
            medium_periods=[(2010, 2015), (2015, 2020), (2020, 2024)],
            longterm_periods=PERIODS,
        )
        # Should have exactly one rate per cell (from one period, no averaging)
        only_recent = three_county_rates[
            (three_county_rates["period_start"] == 2020)
            & (three_county_rates["period_end"] == 2024)
        ]
        expected_cells = only_recent.groupby(["county_fips", "age_group", "sex"]).size()
        assert len(recent) == len(expected_cells)

    def test_longterm_averages_all_five_periods(self, three_county_rates: pd.DataFrame) -> None:
        """Long-term average uses all 5 periods and produces one rate per cell."""
        _, _, longterm = compute_period_window_averages(
            three_county_rates,
            recent_periods=[(2020, 2024)],
            medium_periods=[(2010, 2015), (2015, 2020), (2020, 2024)],
            longterm_periods=PERIODS,
        )
        n_counties = three_county_rates["county_fips"].nunique()
        expected_cells = n_counties * CELLS_PER_COUNTY
        assert len(longterm) == expected_cells


# ---------------------------------------------------------------------------
# Tests: Convergence schedule
# ---------------------------------------------------------------------------


class TestConvergenceSchedule:
    """Tests for calculate_age_specific_convergence."""

    def test_year1_equals_recent_rates(
        self,
        known_rates: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Year 1 should NOT equal recent exactly (t=1/5 interpolation).

        At year=1, t = 1/5 = 0.2, so rate = recent*0.8 + medium*0.2.
        For known_rates: 0.10*0.8 + 0.05*0.2 = 0.08 + 0.01 = 0.09.
        """
        recent, medium, longterm = known_rates
        results = calculate_age_specific_convergence(recent, medium, longterm)
        year1 = results[1]
        expected_rate = 0.10 * (1 - 1 / 5) + 0.05 * (1 / 5)
        np.testing.assert_allclose(
            year1["migration_rate"].values,
            expected_rate,
            atol=1e-10,
        )

    def test_year5_equals_medium_rates(
        self,
        known_rates: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Year 5 (end of phase 1) should equal medium rates exactly.

        At year=5, t = 5/5 = 1.0, so rate = recent*0 + medium*1 = medium.
        """
        recent, medium, longterm = known_rates
        results = calculate_age_specific_convergence(recent, medium, longterm)
        year5 = results[5]
        np.testing.assert_allclose(
            year5["migration_rate"].values,
            0.05,
            atol=1e-10,
        )

    def test_year6_through_15_hold_at_medium(
        self,
        known_rates: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Years 6-15 (phase 2) should hold at medium rate."""
        recent, medium, longterm = known_rates
        results = calculate_age_specific_convergence(recent, medium, longterm)
        for year in range(6, 16):
            year_df = results[year]
            np.testing.assert_allclose(
                year_df["migration_rate"].values,
                0.05,
                atol=1e-10,
                err_msg=f"Year {year} should hold at medium rate",
            )

    def test_year20_equals_longterm_rates(
        self,
        known_rates: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Year 20 (end of phase 3) should equal long-term rates exactly.

        At year=20, years_into_phase3 = 20-5-10 = 5, t = 5/5 = 1.0,
        so rate = medium*0 + longterm*1 = longterm.
        """
        recent, medium, longterm = known_rates
        results = calculate_age_specific_convergence(recent, medium, longterm)
        year20 = results[20]
        np.testing.assert_allclose(
            year20["migration_rate"].values,
            0.02,
            atol=1e-10,
        )

    def test_each_age_group_converges_independently(self) -> None:
        """Different age groups have different rates at every year."""
        cells = [
            {"county_fips": "38001", "age_group": "0-4", "sex": "Male"},
            {"county_fips": "38001", "age_group": "20-24", "sex": "Male"},
        ]
        recent = pd.DataFrame(cells)
        recent["migration_rate"] = [0.10, -0.05]
        medium = pd.DataFrame(cells)
        medium["migration_rate"] = [0.05, 0.00]
        longterm = pd.DataFrame(cells)
        longterm["migration_rate"] = [0.02, 0.02]

        results = calculate_age_specific_convergence(recent, medium, longterm)

        # Check year 1: the two age groups should have different rates
        year1 = results[1]
        rates = year1["migration_rate"].values
        assert rates[0] != rates[1], "Different age groups should converge independently"

        # Check year 10 (medium hold): they should also differ
        year10 = results[10]
        rates10 = year10["migration_rate"].values
        assert rates10[0] != rates10[1]

    def test_positive_and_negative_rates_converge_correctly(
        self,
        mixed_sign_rates: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Both positive and negative rates converge through the schedule."""
        recent, medium, longterm = mixed_sign_rates
        results = calculate_age_specific_convergence(recent, medium, longterm)

        def _get_rate(year_df: pd.DataFrame, sex: str) -> float:
            """Get rate for a specific sex from a year DataFrame."""
            row = year_df[year_df["sex"] == sex]
            return float(row["migration_rate"].iloc[0])

        # Year 5: should be at medium (Male=0.04, Female=-0.03)
        assert _get_rate(results[5], "Male") == pytest.approx(0.04, abs=1e-10)
        assert _get_rate(results[5], "Female") == pytest.approx(-0.03, abs=1e-10)

        # Year 20: should be at long-term (Male=0.01, Female=-0.01)
        assert _get_rate(results[20], "Male") == pytest.approx(0.01, abs=1e-10)
        assert _get_rate(results[20], "Female") == pytest.approx(-0.01, abs=1e-10)

        # Positive rate (Male) should decrease monotonically across phase boundaries
        male_rates = [
            _get_rate(results[1], "Male"),
            _get_rate(results[5], "Male"),
            _get_rate(results[10], "Male"),
            _get_rate(results[20], "Male"),
        ]
        assert male_rates[0] > male_rates[1]
        assert male_rates[1] == male_rates[2]
        assert male_rates[2] > male_rates[3]

        # Negative rate (Female) should increase (toward 0) monotonically
        female_rates = [
            _get_rate(results[1], "Female"),
            _get_rate(results[5], "Female"),
            _get_rate(results[10], "Female"),
            _get_rate(results[20], "Female"),
        ]
        assert female_rates[0] < female_rates[1]
        assert female_rates[1] == female_rates[2]
        assert female_rates[2] < female_rates[3]

    def test_output_shape_per_year(self) -> None:
        """Each year has the correct number of rows (36 per county x n_counties)."""
        n_counties = 53
        counties = ND_COUNTY_FIPS[:n_counties]
        records = [
            {"county_fips": county, "age_group": ag, "sex": sex, "migration_rate": 0.01}
            for county in counties
            for ag in AGE_GROUPS
            for sex in SEX_CATEGORIES
        ]

        recent = pd.DataFrame(records)
        medium = pd.DataFrame(records).assign(migration_rate=0.005)
        longterm = pd.DataFrame(records).assign(migration_rate=0.002)

        results = calculate_age_specific_convergence(recent, medium, longterm)

        expected_per_year = n_counties * CELLS_PER_COUNTY
        assert len(results) == 20
        for year in range(1, 21):
            assert len(results[year]) == expected_per_year, (
                f"Year {year}: expected {expected_per_year}, got {len(results[year])}"
            )

    def test_custom_schedule_parameters(
        self,
        known_rates: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Custom convergence schedule (3-4-3) works correctly."""
        recent, medium, longterm = known_rates

        custom_schedule = {
            "recent_to_medium_years": 3,
            "medium_hold_years": 4,
            "medium_to_longterm_years": 3,
        }
        results = calculate_age_specific_convergence(
            recent,
            medium,
            longterm,
            projection_years=10,
            convergence_schedule=custom_schedule,
        )

        # Year 3 should equal medium (end of phase 1)
        np.testing.assert_allclose(
            results[3]["migration_rate"].values,
            0.05,
            atol=1e-10,
        )

        # Years 4-7 should hold at medium
        for year in [4, 5, 6, 7]:
            np.testing.assert_allclose(
                results[year]["migration_rate"].values,
                0.05,
                atol=1e-10,
                err_msg=f"Year {year} should hold at medium",
            )

        # Year 10 should equal long-term (end of phase 3)
        np.testing.assert_allclose(
            results[10]["migration_rate"].values,
            0.02,
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# Integration test: end-to-end with real data
# ---------------------------------------------------------------------------

_REAL_DATA_PATH = Path(__file__).parent.parent.parent / (
    "data/processed/migration/residual_migration_rates.parquet"
)


@pytest.mark.skipif(
    not _REAL_DATA_PATH.exists(),
    reason="Phase 1 output not available",
)
class TestEndToEndWithRealData:
    """Integration tests using actual Phase 1 output."""

    def test_end_to_end_with_real_data(self) -> None:
        """Full pipeline produces correct output dimensions.

        Expected: 20 years x 53 counties x 36 cells = 38,160 rows.
        """
        all_rates = pd.read_parquet(_REAL_DATA_PATH)

        # Map config windows to periods
        available_periods = sorted(
            all_rates[["period_start", "period_end"]]
            .drop_duplicates()
            .apply(
                lambda r: (int(r["period_start"]), int(r["period_end"])),
                axis=1,
            )
            .tolist()
        )

        recent_periods = _map_config_window_to_periods([2022, 2024], available_periods)
        medium_periods = _map_config_window_to_periods([2014, 2024], available_periods)
        longterm_periods = _map_config_window_to_periods([2000, 2024], available_periods)

        recent, medium, longterm = compute_period_window_averages(
            all_rates,
            recent_periods,
            medium_periods,
            longterm_periods,
        )

        rates_by_year = calculate_age_specific_convergence(
            recent,
            medium,
            longterm,
        )

        # Verify correct number of years
        assert len(rates_by_year) == 20

        # Verify each year has the right number of rows
        n_counties = all_rates["county_fips"].nunique()
        expected_per_year = n_counties * CELLS_PER_COUNTY
        for year_offset, year_df in rates_by_year.items():
            assert len(year_df) == expected_per_year, (
                f"Year {year_offset}: expected {expected_per_year}, got {len(year_df)}"
            )

        # Verify total row count
        total_rows = sum(len(df) for df in rates_by_year.values())
        expected_total = 20 * n_counties * CELLS_PER_COUNTY
        assert total_rows == expected_total, (
            f"Total rows: expected {expected_total}, got {total_rows}"
        )

        # With 53 counties, expect 38,160 rows
        assert total_rows == 38160, f"Expected 38,160 total rows (20*53*36), got {total_rows}"

    def test_year5_matches_medium_average(self) -> None:
        """Year 5 rates match the medium window average for real data."""
        all_rates = pd.read_parquet(_REAL_DATA_PATH)

        available_periods = sorted(
            all_rates[["period_start", "period_end"]]
            .drop_duplicates()
            .apply(
                lambda r: (int(r["period_start"]), int(r["period_end"])),
                axis=1,
            )
            .tolist()
        )

        recent_periods = _map_config_window_to_periods([2022, 2024], available_periods)
        medium_periods = _map_config_window_to_periods([2014, 2024], available_periods)
        longterm_periods = _map_config_window_to_periods([2000, 2024], available_periods)

        recent, medium, longterm = compute_period_window_averages(
            all_rates,
            recent_periods,
            medium_periods,
            longterm_periods,
        )

        rates_by_year = calculate_age_specific_convergence(
            recent,
            medium,
            longterm,
        )

        # Year 5 should equal medium rates
        year5 = (
            rates_by_year[5].sort_values(["county_fips", "age_group", "sex"]).reset_index(drop=True)
        )
        medium_sorted = medium.sort_values(["county_fips", "age_group", "sex"]).reset_index(
            drop=True
        )

        np.testing.assert_allclose(
            year5["migration_rate"].values,
            medium_sorted["migration_rate"].values,
            atol=1e-10,
        )

    def test_year20_matches_longterm_average(self) -> None:
        """Year 20 rates match the long-term window average for real data."""
        all_rates = pd.read_parquet(_REAL_DATA_PATH)

        available_periods = sorted(
            all_rates[["period_start", "period_end"]]
            .drop_duplicates()
            .apply(
                lambda r: (int(r["period_start"]), int(r["period_end"])),
                axis=1,
            )
            .tolist()
        )

        recent_periods = _map_config_window_to_periods([2022, 2024], available_periods)
        medium_periods = _map_config_window_to_periods([2014, 2024], available_periods)
        longterm_periods = _map_config_window_to_periods([2000, 2024], available_periods)

        recent, medium, longterm = compute_period_window_averages(
            all_rates,
            recent_periods,
            medium_periods,
            longterm_periods,
        )

        rates_by_year = calculate_age_specific_convergence(
            recent,
            medium,
            longterm,
        )

        year20 = (
            rates_by_year[20]
            .sort_values(["county_fips", "age_group", "sex"])
            .reset_index(drop=True)
        )
        longterm_sorted = longterm.sort_values(["county_fips", "age_group", "sex"]).reset_index(
            drop=True
        )

        np.testing.assert_allclose(
            year20["migration_rate"].values,
            longterm_sorted["migration_rate"].values,
            atol=1e-10,
        )
