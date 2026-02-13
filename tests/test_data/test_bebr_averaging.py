"""
Unit tests for BEBR multi-period averaging and Census Bureau interpolation.

Tests the pure computation functions added for ADR-036, which implement
BEBR-style multi-period averaging (primary) and Census Bureau convergence
interpolation (secondary) for migration rate calculation.

Uses synthetic data -- no actual PEP data files required.
"""

import pandas as pd
import pytest

from cohort_projections.data.process.migration_rates import (
    DEFAULT_BASE_PERIODS,
    apply_county_dampening,
    calculate_bebr_scenarios,
    calculate_interpolated_rates,
    calculate_multiperiod_averages,
    calculate_period_average,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pep_data(county_patterns: dict[str, dict[int, float]]) -> pd.DataFrame:
    """Create synthetic PEP data from county patterns.

    Args:
        county_patterns: Dict mapping geoid to {year: netmig} pairs.
    """
    records = []
    for geoid, year_values in county_patterns.items():
        for year, netmig in year_values.items():
            records.append({"geoid": geoid, "year": year, "netmig": netmig})
    return pd.DataFrame(records)


def _make_period_avg(county_values: dict[str, float]) -> pd.DataFrame:
    """Create a period average DataFrame from county values.

    Args:
        county_values: Dict mapping geoid to mean_netmig value.
    """
    records = [
        {"geoid": geoid, "mean_netmig": val, "n_years": 5} for geoid, val in county_values.items()
    ]
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# TestCalculatePeriodAverage
# ---------------------------------------------------------------------------


class TestCalculatePeriodAverage:
    """Tests for calculate_period_average function."""

    def test_basic_average(self):
        """1 county, 5 years -- verify mean matches pd.Series.mean()."""
        values = {2020: 100.0, 2021: 200.0, 2022: 300.0, 2023: 400.0, 2024: 500.0}
        pep = _make_pep_data({"38001": values})

        result = calculate_period_average(pep, 2020, 2024)

        expected_mean = pd.Series(list(values.values())).mean()
        assert len(result) == 1
        assert result.iloc[0]["mean_netmig"] == pytest.approx(expected_mean)

    def test_period_filtering(self):
        """PEP data spans 2000-2024; filter to 2019-2024 only."""
        year_values = {yr: float(yr) for yr in range(2000, 2025)}
        pep = _make_pep_data({"38001": year_values})

        result = calculate_period_average(pep, 2019, 2024)

        # Should only include years 2019-2024 (6 years)
        assert result.iloc[0]["n_years"] == 6
        expected_mean = pd.Series([2019.0, 2020.0, 2021.0, 2022.0, 2023.0, 2024.0]).mean()
        assert result.iloc[0]["mean_netmig"] == pytest.approx(expected_mean)

    def test_multi_county(self):
        """3 counties with different netmig -- verify 3 rows returned."""
        pep = _make_pep_data(
            {
                "38001": {2020: 100.0, 2021: 200.0},
                "38003": {2020: -50.0, 2021: -100.0},
                "38017": {2020: 500.0, 2021: 600.0},
            }
        )

        result = calculate_period_average(pep, 2020, 2021)

        assert len(result) == 3
        geoids = set(result["geoid"].tolist())
        assert geoids == {"38001", "38003", "38017"}

    def test_empty_period(self):
        """No data in requested range returns empty DataFrame with correct columns."""
        pep = _make_pep_data({"38001": {2020: 100.0, 2021: 200.0}})

        result = calculate_period_average(pep, 2000, 2005)

        assert result.empty
        assert list(result.columns) == ["geoid", "mean_netmig", "n_years"]

    def test_single_year(self):
        """Period of 1 year -- n_years should be 1, mean = that year's value."""
        pep = _make_pep_data({"38001": {2024: 42.0}})

        result = calculate_period_average(pep, 2024, 2024)

        assert result.iloc[0]["n_years"] == 1
        assert result.iloc[0]["mean_netmig"] == pytest.approx(42.0)

    def test_n_years_count(self):
        """Verify n_years matches actual count of records in the period."""
        # Data has gaps: only 3 of 5 possible years
        pep = _make_pep_data({"38001": {2020: 10.0, 2022: 30.0, 2024: 50.0}})

        result = calculate_period_average(pep, 2020, 2024)

        assert result.iloc[0]["n_years"] == 3


# ---------------------------------------------------------------------------
# TestCalculateMultiperiodAverages
# ---------------------------------------------------------------------------


class TestCalculateMultiperiodAverages:
    """Tests for calculate_multiperiod_averages function."""

    def test_four_standard_periods(self):
        """Use default periods -- verify 4 keys returned."""
        year_values = {yr: float(yr - 2000) * 10 for yr in range(2000, 2025)}
        pep = _make_pep_data({"38001": year_values})

        result = calculate_multiperiod_averages(pep)

        assert set(result.keys()) == {"short", "medium", "long", "full"}
        for df in result.values():
            assert len(df) == 1  # one county
            assert "mean_netmig" in df.columns

    def test_custom_periods(self):
        """Custom 2-period dict -- verify only those 2 returned."""
        year_values = dict.fromkeys(range(2000, 2025), 100.0)
        pep = _make_pep_data({"38001": year_values})

        custom = {
            "early": (2000, 2010),
            "late": (2015, 2024),
        }
        result = calculate_multiperiod_averages(pep, base_periods=custom)

        assert set(result.keys()) == {"early", "late"}

    def test_overlapping_periods(self):
        """Periods that overlap share years; each computed independently."""
        # Years 2018-2024 with ascending values
        year_values = {yr: float(yr) for yr in range(2018, 2025)}
        pep = _make_pep_data({"38001": year_values})

        periods = {
            "a": (2018, 2022),  # 2018,2019,2020,2021,2022
            "b": (2020, 2024),  # 2020,2021,2022,2023,2024
        }
        result = calculate_multiperiod_averages(pep, base_periods=periods)

        mean_a = result["a"].iloc[0]["mean_netmig"]
        mean_b = result["b"].iloc[0]["mean_netmig"]

        expected_a = pd.Series([2018.0, 2019.0, 2020.0, 2021.0, 2022.0]).mean()
        expected_b = pd.Series([2020.0, 2021.0, 2022.0, 2023.0, 2024.0]).mean()

        assert mean_a == pytest.approx(expected_a)
        assert mean_b == pytest.approx(expected_b)
        # The two means should differ because the periods cover different years
        assert mean_a != pytest.approx(mean_b)

    def test_default_base_periods_constant(self):
        """Verify DEFAULT_BASE_PERIODS has expected keys and values."""
        assert set(DEFAULT_BASE_PERIODS.keys()) == {"short", "medium", "long", "full"}
        assert DEFAULT_BASE_PERIODS["short"] == (2019, 2024)
        assert DEFAULT_BASE_PERIODS["medium"] == (2014, 2024)
        assert DEFAULT_BASE_PERIODS["long"] == (2005, 2024)
        assert DEFAULT_BASE_PERIODS["full"] == (2000, 2024)


# ---------------------------------------------------------------------------
# TestCalculateBEBRScenarios
# ---------------------------------------------------------------------------


class TestCalculateBEBRScenarios:
    """Tests for calculate_bebr_scenarios function."""

    def test_trimmed_average_drops_extremes(self):
        """4 periods with values [100, 200, 300, 400].

        Baseline = mean(200, 300) = 250. Low = 100. High = 400.
        """
        period_averages = {
            "p1": _make_period_avg({"38001": 100.0}),
            "p2": _make_period_avg({"38001": 200.0}),
            "p3": _make_period_avg({"38001": 300.0}),
            "p4": _make_period_avg({"38001": 400.0}),
        }

        result = calculate_bebr_scenarios(period_averages)

        baseline = result["baseline"]
        row = baseline[baseline["geoid"] == "38001"].iloc[0]
        assert row["net_migration"] == pytest.approx(250.0)

    def test_low_is_minimum(self):
        """Verify low = min period mean."""
        period_averages = {
            "p1": _make_period_avg({"38001": 100.0}),
            "p2": _make_period_avg({"38001": 200.0}),
            "p3": _make_period_avg({"38001": 300.0}),
            "p4": _make_period_avg({"38001": 400.0}),
        }

        result = calculate_bebr_scenarios(period_averages)

        low = result["low"]
        row = low[low["geoid"] == "38001"].iloc[0]
        assert row["net_migration"] == pytest.approx(100.0)

    def test_high_is_maximum(self):
        """Verify high = max period mean."""
        period_averages = {
            "p1": _make_period_avg({"38001": 100.0}),
            "p2": _make_period_avg({"38001": 200.0}),
            "p3": _make_period_avg({"38001": 300.0}),
            "p4": _make_period_avg({"38001": 400.0}),
        }

        result = calculate_bebr_scenarios(period_averages)

        high = result["high"]
        row = high[high["geoid"] == "38001"].iloc[0]
        assert row["net_migration"] == pytest.approx(400.0)

    def test_all_equal_periods(self):
        """All 4 periods have same value -- baseline = low = high = that value."""
        period_averages = {
            "p1": _make_period_avg({"38001": 150.0}),
            "p2": _make_period_avg({"38001": 150.0}),
            "p3": _make_period_avg({"38001": 150.0}),
            "p4": _make_period_avg({"38001": 150.0}),
        }

        result = calculate_bebr_scenarios(period_averages)

        for scenario in ("baseline", "low", "high"):
            row = result[scenario][result[scenario]["geoid"] == "38001"].iloc[0]
            assert row["net_migration"] == pytest.approx(150.0)

    def test_three_periods(self):
        """3 periods with [100, 200, 300].

        Baseline = 200 (middle). Low = 100. High = 300.
        """
        period_averages = {
            "p1": _make_period_avg({"38001": 100.0}),
            "p2": _make_period_avg({"38001": 200.0}),
            "p3": _make_period_avg({"38001": 300.0}),
        }

        result = calculate_bebr_scenarios(period_averages)

        baseline = result["baseline"]
        row = baseline[baseline["geoid"] == "38001"].iloc[0]
        assert row["net_migration"] == pytest.approx(200.0)

        low_row = result["low"][result["low"]["geoid"] == "38001"].iloc[0]
        assert low_row["net_migration"] == pytest.approx(100.0)

        high_row = result["high"][result["high"]["geoid"] == "38001"].iloc[0]
        assert high_row["net_migration"] == pytest.approx(300.0)

    def test_two_periods(self):
        """2 periods with [100, 300].

        Baseline = 200 (average). Low = 100. High = 300.
        """
        period_averages = {
            "p1": _make_period_avg({"38001": 100.0}),
            "p2": _make_period_avg({"38001": 300.0}),
        }

        result = calculate_bebr_scenarios(period_averages)

        baseline = result["baseline"]
        row = baseline[baseline["geoid"] == "38001"].iloc[0]
        assert row["net_migration"] == pytest.approx(200.0)

        low_row = result["low"][result["low"]["geoid"] == "38001"].iloc[0]
        assert low_row["net_migration"] == pytest.approx(100.0)

        high_row = result["high"][result["high"]["geoid"] == "38001"].iloc[0]
        assert high_row["net_migration"] == pytest.approx(300.0)

    def test_one_period(self):
        """1 period with value 500. All scenarios = 500."""
        period_averages = {
            "only": _make_period_avg({"38001": 500.0}),
        }

        result = calculate_bebr_scenarios(period_averages)

        for scenario in ("baseline", "low", "high"):
            row = result[scenario][result[scenario]["geoid"] == "38001"].iloc[0]
            assert row["net_migration"] == pytest.approx(500.0)

    def test_zero_periods(self):
        """Empty dict -- all scenarios return empty DataFrames."""
        result = calculate_bebr_scenarios({})

        for scenario in ("baseline", "low", "high"):
            assert result[scenario].empty
            assert "geoid" in result[scenario].columns
            assert "net_migration" in result[scenario].columns

    def test_negative_migration(self):
        """County with all-negative periods [-500, -300, -200, -100].

        Baseline = mean(-300, -200) = -250. Low = -500. High = -100.
        """
        period_averages = {
            "p1": _make_period_avg({"38001": -500.0}),
            "p2": _make_period_avg({"38001": -300.0}),
            "p3": _make_period_avg({"38001": -200.0}),
            "p4": _make_period_avg({"38001": -100.0}),
        }

        result = calculate_bebr_scenarios(period_averages)

        baseline = result["baseline"]
        row = baseline[baseline["geoid"] == "38001"].iloc[0]
        assert row["net_migration"] == pytest.approx(-250.0)

        low_row = result["low"][result["low"]["geoid"] == "38001"].iloc[0]
        assert low_row["net_migration"] == pytest.approx(-500.0)

        high_row = result["high"][result["high"]["geoid"] == "38001"].iloc[0]
        assert high_row["net_migration"] == pytest.approx(-100.0)

    def test_mixed_positive_negative(self):
        """Periods with [-200, -50, 100, 400].

        Baseline = mean(-50, 100) = 25. Low = -200. High = 400.
        """
        period_averages = {
            "p1": _make_period_avg({"38001": -200.0}),
            "p2": _make_period_avg({"38001": -50.0}),
            "p3": _make_period_avg({"38001": 100.0}),
            "p4": _make_period_avg({"38001": 400.0}),
        }

        result = calculate_bebr_scenarios(period_averages)

        baseline = result["baseline"]
        row = baseline[baseline["geoid"] == "38001"].iloc[0]
        assert row["net_migration"] == pytest.approx(25.0)

        low_row = result["low"][result["low"]["geoid"] == "38001"].iloc[0]
        assert low_row["net_migration"] == pytest.approx(-200.0)

        high_row = result["high"][result["high"]["geoid"] == "38001"].iloc[0]
        assert high_row["net_migration"] == pytest.approx(400.0)

    def test_multiple_counties(self):
        """2 counties with different patterns -- verify independent calculation."""
        # County A: values [100, 200, 300, 400]
        # County B: values [-100, 0, 100, 200]
        period_averages = {
            "p1": _make_period_avg({"38001": 100.0, "38003": -100.0}),
            "p2": _make_period_avg({"38001": 200.0, "38003": 0.0}),
            "p3": _make_period_avg({"38001": 300.0, "38003": 100.0}),
            "p4": _make_period_avg({"38001": 400.0, "38003": 200.0}),
        }

        result = calculate_bebr_scenarios(period_averages)

        # County A: baseline = mean(200, 300) = 250
        baseline_a = result["baseline"][result["baseline"]["geoid"] == "38001"].iloc[0]
        assert baseline_a["net_migration"] == pytest.approx(250.0)

        # County B: baseline = mean(0, 100) = 50
        baseline_b = result["baseline"][result["baseline"]["geoid"] == "38003"].iloc[0]
        assert baseline_b["net_migration"] == pytest.approx(50.0)

        # County A: low = 100, high = 400
        low_a = result["low"][result["low"]["geoid"] == "38001"].iloc[0]
        assert low_a["net_migration"] == pytest.approx(100.0)
        high_a = result["high"][result["high"]["geoid"] == "38001"].iloc[0]
        assert high_a["net_migration"] == pytest.approx(400.0)

        # County B: low = -100, high = 200
        low_b = result["low"][result["low"]["geoid"] == "38003"].iloc[0]
        assert low_b["net_migration"] == pytest.approx(-100.0)
        high_b = result["high"][result["high"]["geoid"] == "38003"].iloc[0]
        assert high_b["net_migration"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# TestCalculateInterpolatedRates
# ---------------------------------------------------------------------------


class TestCalculateInterpolatedRates:
    """Tests for calculate_interpolated_rates function."""

    @pytest.fixture
    def simple_averages(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Recent=1000, medium=500, longterm=200 for one county."""
        recent = _make_period_avg({"38001": 1000.0})
        medium = _make_period_avg({"38001": 500.0})
        longterm = _make_period_avg({"38001": 200.0})
        return recent, medium, longterm

    def test_year_1_near_recent(self, simple_averages):
        """Year 1: 80% recent + 20% medium (t = 1/5 = 0.2)."""
        recent, medium, longterm = simple_averages

        result = calculate_interpolated_rates(recent, medium, longterm)

        year1 = result[1]
        row = year1[year1["geoid"] == "38001"].iloc[0]
        expected = 1000.0 * 0.8 + 500.0 * 0.2  # = 900
        assert row["net_migration"] == pytest.approx(expected)

    def test_year_5_equals_medium(self, simple_averages):
        """Year 5: 0% recent + 100% medium (t = 5/5 = 1.0)."""
        recent, medium, longterm = simple_averages

        result = calculate_interpolated_rates(recent, medium, longterm)

        year5 = result[5]
        row = year5[year5["geoid"] == "38001"].iloc[0]
        assert row["net_migration"] == pytest.approx(500.0)

    def test_years_6_to_15_constant(self, simple_averages):
        """Years 6-15 all equal to medium average."""
        recent, medium, longterm = simple_averages

        result = calculate_interpolated_rates(recent, medium, longterm)

        for year in range(6, 16):
            year_df = result[year]
            row = year_df[year_df["geoid"] == "38001"].iloc[0]
            assert row["net_migration"] == pytest.approx(500.0), (
                f"Year {year} should equal medium (500.0)"
            )

    def test_year_20_equals_longterm(self, simple_averages):
        """Year 20: 0% medium + 100% longterm."""
        recent, medium, longterm = simple_averages

        result = calculate_interpolated_rates(recent, medium, longterm)

        year20 = result[20]
        row = year20[year20["geoid"] == "38001"].iloc[0]
        assert row["net_migration"] == pytest.approx(200.0)

    def test_year_16_interpolation(self, simple_averages):
        """Year 16: 1/5 through phase 3 -- 80% medium + 20% longterm."""
        recent, medium, longterm = simple_averages

        result = calculate_interpolated_rates(recent, medium, longterm)

        year16 = result[16]
        row = year16[year16["geoid"] == "38001"].iloc[0]
        # t = (16 - 5 - 10) / 5 = 1/5 = 0.2
        expected = 500.0 * 0.8 + 200.0 * 0.2  # = 440
        assert row["net_migration"] == pytest.approx(expected)

    def test_correct_year_count(self, simple_averages):
        """Default 20 years returns dict with keys 1-20."""
        recent, medium, longterm = simple_averages

        result = calculate_interpolated_rates(recent, medium, longterm)

        assert set(result.keys()) == set(range(1, 21))
        assert len(result) == 20

    def test_custom_schedule(self):
        """Non-default convergence schedule (3+7+5 = 15 total years)."""
        recent = _make_period_avg({"38001": 1000.0})
        medium = _make_period_avg({"38001": 400.0})
        longterm = _make_period_avg({"38001": 100.0})

        schedule = {
            "recent_to_medium_years": 3,
            "medium_hold_years": 7,
            "medium_to_longterm_years": 5,
        }
        result = calculate_interpolated_rates(
            recent,
            medium,
            longterm,
            projection_years=15,
            convergence_schedule=schedule,
        )

        assert set(result.keys()) == set(range(1, 16))

        # Year 1: t = 1/3 -> (1 - 1/3)*1000 + (1/3)*400 = 800
        y1 = result[1][result[1]["geoid"] == "38001"].iloc[0]
        expected_y1 = 1000.0 * (1 - 1 / 3) + 400.0 * (1 / 3)
        assert y1["net_migration"] == pytest.approx(expected_y1)

        # Year 3: fully medium -> 400
        y3 = result[3][result[3]["geoid"] == "38001"].iloc[0]
        assert y3["net_migration"] == pytest.approx(400.0)

        # Years 4-10: hold at medium -> 400
        for yr in range(4, 11):
            row = result[yr][result[yr]["geoid"] == "38001"].iloc[0]
            assert row["net_migration"] == pytest.approx(400.0), (
                f"Year {yr} should equal medium (400.0)"
            )

        # Year 15: fully longterm -> 100
        y15 = result[15][result[15]["geoid"] == "38001"].iloc[0]
        assert y15["net_migration"] == pytest.approx(100.0)

    def test_single_county(self):
        """Single county -- verify interpolation values at boundary years."""
        recent = _make_period_avg({"38017": 600.0})
        medium = _make_period_avg({"38017": 300.0})
        longterm = _make_period_avg({"38017": 0.0})

        result = calculate_interpolated_rates(recent, medium, longterm)

        # Year 1: t=0.2 -> 600*0.8 + 300*0.2 = 540
        y1 = result[1][result[1]["geoid"] == "38017"].iloc[0]
        assert y1["net_migration"] == pytest.approx(540.0)

        # Year 5: fully medium -> 300
        y5 = result[5][result[5]["geoid"] == "38017"].iloc[0]
        assert y5["net_migration"] == pytest.approx(300.0)

        # Year 10 (mid hold): medium -> 300
        y10 = result[10][result[10]["geoid"] == "38017"].iloc[0]
        assert y10["net_migration"] == pytest.approx(300.0)

        # Year 15 (end of hold): medium -> 300
        y15 = result[15][result[15]["geoid"] == "38017"].iloc[0]
        assert y15["net_migration"] == pytest.approx(300.0)

        # Year 20: fully longterm -> 0
        y20 = result[20][result[20]["geoid"] == "38017"].iloc[0]
        assert y20["net_migration"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestApplyCountyDampening
# ---------------------------------------------------------------------------


def _make_bebr_scenarios(
    county_values: dict[str, dict[str, float]],
) -> dict[str, pd.DataFrame]:
    """Create BEBR scenario dicts from county values.

    Args:
        county_values: Dict mapping geoid to {scenario: net_migration}.
            e.g. {"38105": {"baseline": 1000, "low": 500, "high": 1500}}
    """
    scenarios: dict[str, list[dict]] = {"baseline": [], "low": [], "high": []}
    for geoid, scenario_vals in county_values.items():
        for scenario_name in scenarios:
            scenarios[scenario_name].append(
                {"geoid": geoid, "net_migration": scenario_vals.get(scenario_name, 0.0)}
            )
    return {name: pd.DataFrame(records) for name, records in scenarios.items()}


class TestApplyCountyDampening:
    """Tests for apply_county_dampening function."""

    def test_dampening_reduces_designated_counties(self):
        """Dampened county's net_migration is multiplied by factor; other county unchanged."""
        bebr = _make_bebr_scenarios(
            {
                "38105": {"baseline": 1000.0, "low": 500.0, "high": 1500.0},
                "38017": {"baseline": 800.0, "low": 400.0, "high": 1200.0},
            }
        )

        config = {
            "enabled": True,
            "factor": 0.60,
            "counties": ["38105"],
        }

        result = apply_county_dampening(bebr, config)

        # Dampened county should be 1000 * 0.60 = 600
        dampened = result["baseline"][result["baseline"]["geoid"] == "38105"].iloc[0]
        assert dampened["net_migration"] == pytest.approx(600.0)

        # Non-dampened county should remain unchanged
        unchanged = result["baseline"][result["baseline"]["geoid"] == "38017"].iloc[0]
        assert unchanged["net_migration"] == pytest.approx(800.0)

    def test_dampening_disabled_returns_unchanged(self):
        """With enabled: False, output values should equal input values."""
        bebr = _make_bebr_scenarios(
            {
                "38105": {"baseline": 1000.0, "low": 500.0, "high": 1500.0},
            }
        )

        config = {
            "enabled": False,
            "factor": 0.60,
            "counties": ["38105"],
        }

        result = apply_county_dampening(bebr, config)

        row = result["baseline"][result["baseline"]["geoid"] == "38105"].iloc[0]
        assert row["net_migration"] == pytest.approx(1000.0)

    def test_dampening_none_config_returns_unchanged(self):
        """With dampening_config=None, output should be unchanged."""
        bebr = _make_bebr_scenarios(
            {
                "38105": {"baseline": 1000.0, "low": 500.0, "high": 1500.0},
            }
        )

        result = apply_county_dampening(bebr, None)

        row = result["baseline"][result["baseline"]["geoid"] == "38105"].iloc[0]
        assert row["net_migration"] == pytest.approx(1000.0)

    def test_dampening_applies_to_all_scenarios(self):
        """Verify baseline, low, and high are all dampened."""
        bebr = _make_bebr_scenarios(
            {
                "38105": {"baseline": 1000.0, "low": 500.0, "high": 1500.0},
            }
        )

        config = {
            "enabled": True,
            "factor": 0.60,
            "counties": ["38105"],
        }

        result = apply_county_dampening(bebr, config)

        baseline_val = result["baseline"][result["baseline"]["geoid"] == "38105"].iloc[0]
        assert baseline_val["net_migration"] == pytest.approx(600.0)

        low_val = result["low"][result["low"]["geoid"] == "38105"].iloc[0]
        assert low_val["net_migration"] == pytest.approx(300.0)

        high_val = result["high"][result["high"]["geoid"] == "38105"].iloc[0]
        assert high_val["net_migration"] == pytest.approx(900.0)

    def test_dampening_does_not_mutate_input(self):
        """Verify original dict is not modified."""
        bebr = _make_bebr_scenarios(
            {
                "38105": {"baseline": 1000.0, "low": 500.0, "high": 1500.0},
            }
        )

        # Save original values
        original_baseline = bebr["baseline"]["net_migration"].iloc[0]

        config = {
            "enabled": True,
            "factor": 0.60,
            "counties": ["38105"],
        }

        _ = apply_county_dampening(bebr, config)

        # Original should be unchanged
        assert bebr["baseline"]["net_migration"].iloc[0] == pytest.approx(original_baseline)
        assert bebr["baseline"]["net_migration"].iloc[0] == pytest.approx(1000.0)

    def test_dampening_with_negative_migration(self):
        """Dampening works correctly with negative values.

        0.60 * -100 = -60, reducing the magnitude of out-migration.
        """
        bebr = _make_bebr_scenarios(
            {
                "38105": {"baseline": -100.0, "low": -200.0, "high": -50.0},
            }
        )

        config = {
            "enabled": True,
            "factor": 0.60,
            "counties": ["38105"],
        }

        result = apply_county_dampening(bebr, config)

        baseline_val = result["baseline"][result["baseline"]["geoid"] == "38105"].iloc[0]
        assert baseline_val["net_migration"] == pytest.approx(-60.0)

        low_val = result["low"][result["low"]["geoid"] == "38105"].iloc[0]
        assert low_val["net_migration"] == pytest.approx(-120.0)

        high_val = result["high"][result["high"]["geoid"] == "38105"].iloc[0]
        assert high_val["net_migration"] == pytest.approx(-30.0)

    def test_dampening_empty_county_list(self):
        """Empty counties list results in no changes."""
        bebr = _make_bebr_scenarios(
            {
                "38105": {"baseline": 1000.0, "low": 500.0, "high": 1500.0},
            }
        )

        config = {
            "enabled": True,
            "factor": 0.60,
            "counties": [],
        }

        result = apply_county_dampening(bebr, config)

        row = result["baseline"][result["baseline"]["geoid"] == "38105"].iloc[0]
        assert row["net_migration"] == pytest.approx(1000.0)
