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
    _apply_rate_cap,
    _lift_window_averages,
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
# Tests: Rate cap (ADR-043)
# ---------------------------------------------------------------------------


class TestRateCap:
    """Tests for _apply_rate_cap and rate cap integration."""

    def test_apply_rate_cap_clips_extreme_positive(self) -> None:
        """General cap clips rates above +8% for non-college ages."""
        rate = pd.Series([0.12, 0.05, -0.03, 0.09])
        age_groups = pd.Series(["25-29", "30-34", "35-39", "40-44"])
        config = {
            "college_ages": ["15-19", "20-24"],
            "college_cap": 0.15,
            "general_cap": 0.08,
        }

        capped, n_clipped = _apply_rate_cap(rate, age_groups, config)

        assert capped.iloc[0] == pytest.approx(0.08)
        assert capped.iloc[1] == pytest.approx(0.05)
        assert capped.iloc[2] == pytest.approx(-0.03)
        assert capped.iloc[3] == pytest.approx(0.08)
        assert n_clipped == 2

    def test_apply_rate_cap_clips_extreme_negative(self) -> None:
        """General cap clips rates below -8% for non-college ages."""
        rate = pd.Series([-0.12, -0.05, 0.03, -0.15])
        age_groups = pd.Series(["85+", "70-74", "65-69", "80-84"])
        config = {
            "college_ages": ["15-19", "20-24"],
            "college_cap": 0.15,
            "general_cap": 0.08,
        }

        capped, n_clipped = _apply_rate_cap(rate, age_groups, config)

        assert capped.iloc[0] == pytest.approx(-0.08)
        assert capped.iloc[1] == pytest.approx(-0.05)
        assert capped.iloc[2] == pytest.approx(0.03)
        assert capped.iloc[3] == pytest.approx(-0.08)
        assert n_clipped == 2

    def test_apply_rate_cap_college_ages_use_wider_cap(self) -> None:
        """College-age cells use +/-15% cap instead of +/-8%."""
        rate = pd.Series([0.12, 0.14, -0.10, -0.16])
        age_groups = pd.Series(["15-19", "20-24", "15-19", "20-24"])
        config = {
            "college_ages": ["15-19", "20-24"],
            "college_cap": 0.15,
            "general_cap": 0.08,
        }

        capped, n_clipped = _apply_rate_cap(rate, age_groups, config)

        # 12% and 14% are within +/-15%, should be unchanged
        assert capped.iloc[0] == pytest.approx(0.12)
        assert capped.iloc[1] == pytest.approx(0.14)
        # -10% is within +/-15%, should be unchanged
        assert capped.iloc[2] == pytest.approx(-0.10)
        # -16% exceeds -15%, should be clipped
        assert capped.iloc[3] == pytest.approx(-0.15)
        assert n_clipped == 1

    def test_apply_rate_cap_mixed_ages(self) -> None:
        """Mixed college and non-college ages apply different caps."""
        rate = pd.Series([0.12, 0.12, -0.10, -0.10])
        age_groups = pd.Series(["20-24", "25-29", "15-19", "85+"])
        config = {
            "college_ages": ["15-19", "20-24"],
            "college_cap": 0.15,
            "general_cap": 0.08,
        }

        capped, n_clipped = _apply_rate_cap(rate, age_groups, config)

        # 20-24 at +12%: within +/-15% college cap, unchanged
        assert capped.iloc[0] == pytest.approx(0.12)
        # 25-29 at +12%: exceeds +8% general cap, clipped to 8%
        assert capped.iloc[1] == pytest.approx(0.08)
        # 15-19 at -10%: within +/-15% college cap, unchanged
        assert capped.iloc[2] == pytest.approx(-0.10)
        # 85+ at -10%: exceeds -8% general cap, clipped to -8%
        assert capped.iloc[3] == pytest.approx(-0.08)
        assert n_clipped == 2

    def test_apply_rate_cap_no_clipping_when_within_bounds(self) -> None:
        """No cells clipped when all rates are within bounds."""
        rate = pd.Series([0.05, -0.03, 0.07, -0.06])
        age_groups = pd.Series(["0-4", "5-9", "10-14", "25-29"])
        config = {
            "college_ages": ["15-19", "20-24"],
            "college_cap": 0.15,
            "general_cap": 0.08,
        }

        capped, n_clipped = _apply_rate_cap(rate, age_groups, config)

        pd.testing.assert_series_equal(capped, rate)
        assert n_clipped == 0

    def test_convergence_with_rate_cap_enabled(self) -> None:
        """Rate cap is applied when config has enabled=True."""
        cells = [
            {"county_fips": "38001", "age_group": "25-29", "sex": "Male"},
            {"county_fips": "38001", "age_group": "20-24", "sex": "Female"},
        ]
        # Set all windows to high rates so cap is exercised
        recent = pd.DataFrame(cells)
        recent["migration_rate"] = [0.20, 0.20]
        medium = pd.DataFrame(cells)
        medium["migration_rate"] = [0.20, 0.20]
        longterm = pd.DataFrame(cells)
        longterm["migration_rate"] = [0.20, 0.20]

        rate_cap_config = {
            "enabled": True,
            "college_ages": ["15-19", "20-24"],
            "college_cap": 0.15,
            "general_cap": 0.08,
        }

        results = calculate_age_specific_convergence(
            recent, medium, longterm,
            rate_cap_config=rate_cap_config,
        )

        # Check year 10 (medium hold at 20%)
        year10 = results[10]
        rate_25_29 = float(
            year10.loc[year10["age_group"] == "25-29", "migration_rate"].iloc[0]
        )
        rate_20_24 = float(
            year10.loc[year10["age_group"] == "20-24", "migration_rate"].iloc[0]
        )

        # 25-29 at 20% should be clipped to general_cap=8%
        assert rate_25_29 == pytest.approx(0.08)
        # 20-24 at 20% should be clipped to college_cap=15%
        assert rate_20_24 == pytest.approx(0.15)

    def test_convergence_without_rate_cap(self) -> None:
        """Rate cap does not apply when config is None."""
        cells = [
            {"county_fips": "38001", "age_group": "25-29", "sex": "Male"},
        ]
        recent = pd.DataFrame(cells)
        recent["migration_rate"] = [0.20]
        medium = pd.DataFrame(cells)
        medium["migration_rate"] = [0.20]
        longterm = pd.DataFrame(cells)
        longterm["migration_rate"] = [0.20]

        results = calculate_age_specific_convergence(
            recent, medium, longterm,
            rate_cap_config=None,
        )

        year10 = results[10]
        rate_val = float(year10["migration_rate"].iloc[0])
        # Without cap, rate should remain at 20%
        assert rate_val == pytest.approx(0.20)

    def test_convergence_with_rate_cap_disabled(self) -> None:
        """Rate cap does not apply when enabled=False."""
        cells = [
            {"county_fips": "38001", "age_group": "25-29", "sex": "Male"},
        ]
        recent = pd.DataFrame(cells)
        recent["migration_rate"] = [0.20]
        medium = pd.DataFrame(cells)
        medium["migration_rate"] = [0.20]
        longterm = pd.DataFrame(cells)
        longterm["migration_rate"] = [0.20]

        rate_cap_config = {
            "enabled": False,
            "college_ages": ["15-19", "20-24"],
            "college_cap": 0.15,
            "general_cap": 0.08,
        }

        results = calculate_age_specific_convergence(
            recent, medium, longterm,
            rate_cap_config=rate_cap_config,
        )

        year10 = results[10]
        rate_val = float(year10["migration_rate"].iloc[0])
        # With cap disabled, rate should remain at 20%
        assert rate_val == pytest.approx(0.20)

    def test_rate_cap_preserves_convergence_schedule(self) -> None:
        """Convergence schedule still works correctly with cap enabled.

        Moderate rates (within cap bounds) should still follow the
        5-10-5 interpolation schedule exactly.
        """
        cells = [
            {"county_fips": "38001", "age_group": "0-4", "sex": "Male"},
        ]
        recent = pd.DataFrame(cells)
        recent["migration_rate"] = [0.06]  # within 8% cap
        medium = pd.DataFrame(cells)
        medium["migration_rate"] = [0.03]  # within 8% cap
        longterm = pd.DataFrame(cells)
        longterm["migration_rate"] = [0.01]  # within 8% cap

        rate_cap_config = {
            "enabled": True,
            "college_ages": ["15-19", "20-24"],
            "college_cap": 0.15,
            "general_cap": 0.08,
        }

        results = calculate_age_specific_convergence(
            recent, medium, longterm,
            rate_cap_config=rate_cap_config,
        )

        # Year 1: t=1/5, rate = 0.06*0.8 + 0.03*0.2 = 0.054
        assert float(results[1]["migration_rate"].iloc[0]) == pytest.approx(0.054, abs=1e-10)
        # Year 5: medium = 0.03
        assert float(results[5]["migration_rate"].iloc[0]) == pytest.approx(0.03, abs=1e-10)
        # Year 10: medium hold = 0.03
        assert float(results[10]["migration_rate"].iloc[0]) == pytest.approx(0.03, abs=1e-10)
        # Year 20: longterm = 0.01
        assert float(results[20]["migration_rate"].iloc[0]) == pytest.approx(0.01, abs=1e-10)


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


# ---------------------------------------------------------------------------
# Tests: _lift_window_averages (ADR-046)
# ---------------------------------------------------------------------------


class TestLiftWindowAverages:
    """Tests for _lift_window_averages helper function."""

    def test_lift_adds_increment_to_all_cells(self) -> None:
        """Additive increment is applied to every cell."""
        cells = [
            {"county_fips": "38001", "age_group": "0-4", "sex": "Male"},
            {"county_fips": "38001", "age_group": "0-4", "sex": "Female"},
            {"county_fips": "38017", "age_group": "0-4", "sex": "Male"},
            {"county_fips": "38017", "age_group": "0-4", "sex": "Female"},
        ]
        rates = pd.DataFrame(cells)
        rates["migration_rate"] = [-0.02, -0.01, 0.03, 0.01]

        increment = pd.DataFrame(cells)
        increment["rate_increment"] = [0.005, 0.005, 0.003, 0.003]

        result = _lift_window_averages(rates, increment)

        expected = [-0.015, -0.005, 0.033, 0.013]
        np.testing.assert_allclose(
            result["migration_rate"].values,
            expected,
            atol=1e-10,
        )

    def test_lift_with_zero_increment_is_identity(self) -> None:
        """Zero increment leaves rates unchanged."""
        cells = [
            {"county_fips": "38001", "age_group": "0-4", "sex": "Male"},
        ]
        rates = pd.DataFrame(cells)
        rates["migration_rate"] = [-0.02]

        increment = pd.DataFrame(cells)
        increment["rate_increment"] = [0.0]

        result = _lift_window_averages(rates, increment)
        assert result["migration_rate"].iloc[0] == pytest.approx(-0.02)

    def test_lift_with_missing_county_uses_zero(self) -> None:
        """Counties not in the increment DataFrame get zero increment."""
        rates = pd.DataFrame(
            [
                {"county_fips": "38001", "age_group": "0-4", "sex": "Male", "migration_rate": -0.02},
                {"county_fips": "38099", "age_group": "0-4", "sex": "Male", "migration_rate": 0.01},
            ]
        )
        increment = pd.DataFrame(
            [
                {"county_fips": "38001", "age_group": "0-4", "sex": "Male", "rate_increment": 0.01},
            ]
        )

        result = _lift_window_averages(rates, increment)
        # 38001 should be lifted
        assert result.iloc[0]["migration_rate"] == pytest.approx(-0.01)
        # 38099 should be unchanged (no increment row)
        assert result.iloc[1]["migration_rate"] == pytest.approx(0.01)

    def test_lift_guarantees_higher_rates_with_positive_increment(self) -> None:
        """A positive increment always produces rates >= original."""
        np.random.seed(42)
        n_cells = 100
        rates = pd.DataFrame({
            "county_fips": [f"38{i:03d}" for i in range(n_cells)],
            "age_group": ["0-4"] * n_cells,
            "sex": ["Male"] * n_cells,
            "migration_rate": np.random.uniform(-0.10, 0.10, n_cells),
        })
        increment = pd.DataFrame({
            "county_fips": rates["county_fips"],
            "age_group": rates["age_group"],
            "sex": rates["sex"],
            "rate_increment": np.abs(np.random.uniform(0.001, 0.01, n_cells)),
        })

        result = _lift_window_averages(rates, increment)
        assert (result["migration_rate"].values >= rates["migration_rate"].values).all()


# ---------------------------------------------------------------------------
# Integration tests: High-scenario convergence rates (ADR-046)
# ---------------------------------------------------------------------------

_HIGH_CONVERGENCE_PATH = Path(__file__).parent.parent.parent / (
    "data/processed/migration/convergence_rates_by_year_high.parquet"
)
_BASELINE_CONVERGENCE_PATH = Path(__file__).parent.parent.parent / (
    "data/processed/migration/convergence_rates_by_year.parquet"
)


@pytest.mark.skipif(
    not _HIGH_CONVERGENCE_PATH.exists() or not _BASELINE_CONVERGENCE_PATH.exists(),
    reason="Convergence rate files not available",
)
class TestHighScenarioConvergenceRates:
    """Integration tests for high-scenario convergence rates (ADR-046)."""

    def test_high_and_baseline_same_shape(self) -> None:
        """High and baseline convergence files have identical structure."""
        baseline = pd.read_parquet(_BASELINE_CONVERGENCE_PATH)
        high = pd.read_parquet(_HIGH_CONVERGENCE_PATH)

        assert baseline.shape == high.shape, (
            f"Shape mismatch: baseline {baseline.shape} vs high {high.shape}"
        )
        assert list(baseline.columns) == list(high.columns)

    def test_high_and_baseline_same_counties(self) -> None:
        """Both files cover the same 53 counties."""
        baseline = pd.read_parquet(_BASELINE_CONVERGENCE_PATH)
        high = pd.read_parquet(_HIGH_CONVERGENCE_PATH)

        bl_counties = sorted(baseline["county_fips"].unique())
        hi_counties = sorted(high["county_fips"].unique())

        assert bl_counties == hi_counties
        assert len(bl_counties) == 53

    def test_high_greater_equal_baseline_all_cells(self) -> None:
        """High-scenario rates >= baseline for every cell (ADR-046 guarantee)."""
        baseline = pd.read_parquet(_BASELINE_CONVERGENCE_PATH)
        high = pd.read_parquet(_HIGH_CONVERGENCE_PATH)

        # Sort both identically to align rows
        sort_cols = ["county_fips", "year_offset", "age_group", "sex"]
        baseline_sorted = baseline.sort_values(sort_cols).reset_index(drop=True)
        high_sorted = high.sort_values(sort_cols).reset_index(drop=True)

        violations = (
            high_sorted["migration_rate"].values
            < baseline_sorted["migration_rate"].values - 1e-12
        )
        n_violations = violations.sum()
        assert n_violations == 0, (
            f"{n_violations} cells where high < baseline"
        )

    def test_high_strictly_greater_for_most_cells(self) -> None:
        """Most cells should be strictly greater (not just equal at cap)."""
        baseline = pd.read_parquet(_BASELINE_CONVERGENCE_PATH)
        high = pd.read_parquet(_HIGH_CONVERGENCE_PATH)

        sort_cols = ["county_fips", "year_offset", "age_group", "sex"]
        baseline_sorted = baseline.sort_values(sort_cols).reset_index(drop=True)
        high_sorted = high.sort_values(sort_cols).reset_index(drop=True)

        strictly_greater = (
            high_sorted["migration_rate"].values
            > baseline_sorted["migration_rate"].values + 1e-12
        )
        pct_strictly_greater = strictly_greater.sum() / len(high_sorted) * 100

        # At least 90% of cells should be strictly greater
        assert pct_strictly_greater > 90.0, (
            f"Only {pct_strictly_greater:.1f}% strictly greater (expected >90%)"
        )

    def test_every_county_higher_at_every_year(self) -> None:
        """For each county at each year offset, high mean rate >= baseline mean rate."""
        baseline = pd.read_parquet(_BASELINE_CONVERGENCE_PATH)
        high = pd.read_parquet(_HIGH_CONVERGENCE_PATH)

        bl_means = baseline.groupby(
            ["county_fips", "year_offset"]
        )["migration_rate"].mean()
        hi_means = high.groupby(
            ["county_fips", "year_offset"]
        )["migration_rate"].mean()

        # Align on same multi-index
        comparison = pd.DataFrame({
            "baseline": bl_means,
            "high": hi_means,
        }).dropna()

        violations = comparison[comparison["high"] < comparison["baseline"] - 1e-12]
        assert violations.empty, (
            f"{len(violations)} county-year combinations where high mean < baseline mean:\n"
            f"{violations.head(10)}"
        )

    def test_expected_row_count(self) -> None:
        """Both files should have 57,240 rows (30 years x 53 counties x 36 cells)."""
        baseline = pd.read_parquet(_BASELINE_CONVERGENCE_PATH)
        high = pd.read_parquet(_HIGH_CONVERGENCE_PATH)

        expected = 30 * 53 * 36  # 57,240
        assert len(baseline) == expected, f"Baseline has {len(baseline)} rows, expected {expected}"
        assert len(high) == expected, f"High has {len(high)} rows, expected {expected}"


# ---------------------------------------------------------------------------
# Tests: +15_percent scenario removed (ADR-046)
# ---------------------------------------------------------------------------


class TestPlusPercentRemoved:
    """Verify that the +15_percent migration scenario handler is removed."""

    def test_plus_15_percent_not_in_migration_module(self) -> None:
        """The +15_percent string should not appear as a scenario handler."""
        from cohort_projections.core.migration import apply_migration_scenario

        # Create test migration rates
        rates = pd.DataFrame({
            "age": [20, 30],
            "sex": ["Male", "Male"],
            "race": ["White", "White"],
            "net_migration": [100.0, -50.0],
        })

        # Applying "+15_percent" should trigger the unknown scenario path
        # and return unchanged rates (like "constant" / "recent_average")
        result = apply_migration_scenario(rates, "+15_percent", year=2030, base_year=2025)
        pd.testing.assert_frame_equal(result, rates)
