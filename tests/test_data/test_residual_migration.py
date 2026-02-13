"""
Tests for residual migration rate computation.

Uses synthetic data fixtures for unit tests.  Integration tests use actual
data files (skipped if files are not available).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cohort_projections.data.load.census_age_sex_population import AGE_GROUP_LABELS
from cohort_projections.data.process.residual_migration import (
    apply_college_age_adjustment,
    apply_male_migration_dampening,
    apply_period_dampening,
    average_period_rates,
    compute_residual_migration_rates,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def age_groups():
    """Standard 18 five-year age groups."""
    return AGE_GROUP_LABELS


@pytest.fixture
def survival_rates():
    """Synthetic survival rates for all 18 age groups x 2 sexes."""
    records = []
    for ag in AGE_GROUP_LABELS:
        for sex in ["Male", "Female"]:
            # Realistic-ish rates: high for young, declining for old
            idx = AGE_GROUP_LABELS.index(ag)
            rate = max(0.5, 1.0 - idx * 0.02)
            if ag == "85+":
                rate = 0.60
            records.append({"age_group": ag, "sex": sex, "survival_rate_5yr": rate})
    return pd.DataFrame(records)


def _make_population(
    county_fips: str,
    base: float = 100.0,
) -> pd.DataFrame:
    """Create a synthetic population DataFrame for one county."""
    records = [
        {
            "county_fips": county_fips,
            "age_group": ag,
            "sex": sex,
            "population": base,
        }
        for ag in AGE_GROUP_LABELS
        for sex in ["Male", "Female"]
    ]
    return pd.DataFrame(records)


@pytest.fixture
def pop_start():
    """Synthetic starting population for 2 counties."""
    c1 = _make_population("38001", base=1000.0)
    c2 = _make_population("38017", base=5000.0)
    return pd.concat([c1, c2], ignore_index=True)


@pytest.fixture
def pop_end():
    """Synthetic ending population (slightly different from start)."""
    c1 = _make_population("38001", base=1050.0)  # Growth county
    c2 = _make_population("38017", base=4900.0)  # Decline county
    return pd.concat([c1, c2], ignore_index=True)


# ---------------------------------------------------------------------------
# TestComputeResidualMigrationRates
# ---------------------------------------------------------------------------


class TestComputeResidualMigrationRates:
    """Tests for the core residual migration computation."""

    def test_basic_computation(self, pop_start, pop_end, survival_rates):
        """Residual migration produces correct columns and row count."""
        result = compute_residual_migration_rates(
            pop_start, pop_end, survival_rates, period=(2000, 2005)
        )
        expected_cols = {
            "county_fips",
            "age_group",
            "sex",
            "period_start",
            "period_end",
            "pop_start",
            "expected_pop",
            "pop_end",
            "net_migration",
            "migration_rate",
        }
        assert expected_cols.issubset(set(result.columns))

        # 2 counties * 18 age groups * 2 sexes = 72
        assert len(result) == 2 * 18 * 2

    def test_age_shifting(self, survival_rates):
        """Population at age 0-4 contributes to age 5-9 at end of period."""
        pop_s = pd.DataFrame(
            [
                {
                    "county_fips": "38001",
                    "age_group": ag,
                    "sex": "Male",
                    "population": 100.0 if ag == "0-4" else 0.0,
                }
                for ag in AGE_GROUP_LABELS
            ]
        )
        pop_e = pd.DataFrame(
            [
                {
                    "county_fips": "38001",
                    "age_group": ag,
                    "sex": "Male",
                    "population": 120.0 if ag == "5-9" else 0.0,
                }
                for ag in AGE_GROUP_LABELS
            ]
        )

        result = compute_residual_migration_rates(pop_s, pop_e, survival_rates, period=(2000, 2005))

        row_5_9 = result[(result["age_group"] == "5-9") & (result["sex"] == "Male")].iloc[0]

        # Expected = 100 * survival_rate for 0-4 Male
        surv_0_4 = survival_rates[
            (survival_rates["age_group"] == "0-4") & (survival_rates["sex"] == "Male")
        ]["survival_rate_5yr"].values[0]

        expected = 100.0 * surv_0_4
        np.testing.assert_allclose(row_5_9["expected_pop"], expected, rtol=1e-6)
        np.testing.assert_allclose(row_5_9["net_migration"], 120.0 - expected, rtol=1e-6)

    def test_85plus_open_ended(self, survival_rates):
        """85+ group combines survived 80-84 and survived 85+."""
        pop_s = pd.DataFrame(
            [
                {
                    "county_fips": "38001",
                    "age_group": ag,
                    "sex": "Female",
                    "population": 200.0 if ag in ("80-84", "85+") else 0.0,
                }
                for ag in AGE_GROUP_LABELS
            ]
        )
        pop_e = pd.DataFrame(
            [
                {
                    "county_fips": "38001",
                    "age_group": ag,
                    "sex": "Female",
                    "population": 300.0 if ag == "85+" else 0.0,
                }
                for ag in AGE_GROUP_LABELS
            ]
        )

        result = compute_residual_migration_rates(pop_s, pop_e, survival_rates, period=(2000, 2005))

        row_85 = result[
            (result["county_fips"] == "38001")
            & (result["age_group"] == "85+")
            & (result["sex"] == "Female")
        ].iloc[0]

        surv_80 = survival_rates[
            (survival_rates["age_group"] == "80-84") & (survival_rates["sex"] == "Female")
        ]["survival_rate_5yr"].values[0]
        surv_85 = survival_rates[
            (survival_rates["age_group"] == "85+") & (survival_rates["sex"] == "Female")
        ]["survival_rate_5yr"].values[0]

        expected_85 = 200.0 * surv_80 + 200.0 * surv_85
        np.testing.assert_allclose(row_85["expected_pop"], expected_85, rtol=1e-6)
        np.testing.assert_allclose(row_85["net_migration"], 300.0 - expected_85, rtol=1e-6)

    def test_negative_migration_rates(self, survival_rates):
        """Net out-migration produces negative rates."""
        pop_s = _make_population("38001", base=1000.0)
        # End population much smaller -> out-migration
        pop_e = _make_population("38001", base=500.0)

        result = compute_residual_migration_rates(pop_s, pop_e, survival_rates, period=(2000, 2005))

        # Most shifted age groups should have negative migration
        shifted_rows = result[~result["age_group"].isin(["0-4"]) & (result["expected_pop"] > 0)]
        assert (shifted_rows["net_migration"] < 0).any()

    def test_zero_starting_population(self, survival_rates):
        """Zero starting population gives zero migration rate."""
        pop_s = _make_population("38001", base=0.0)
        pop_e = _make_population("38001", base=100.0)

        result = compute_residual_migration_rates(pop_s, pop_e, survival_rates, period=(2000, 2005))

        # When pop_start is 0, expected is 0, rate should be 0
        shifted = result[result["age_group"] != "0-4"]
        # 85+ gets special handling but with 0 start pop, expected is still 0
        assert (shifted["migration_rate"] == 0.0).all()

    def test_four_year_period_adjustment(self, pop_start, pop_end, survival_rates):
        """4-year period adjusts survival rates by exponent 4/5."""
        result = compute_residual_migration_rates(
            pop_start, pop_end, survival_rates, period=(2020, 2024)
        )

        # Verify that expected pop differs from 5-year calculation
        result_5yr = compute_residual_migration_rates(
            pop_start, pop_end, survival_rates, period=(2000, 2005)
        )

        # Expected pop should be higher for 4-year (less mortality)
        row_4yr = result[
            (result["county_fips"] == "38001")
            & (result["age_group"] == "5-9")
            & (result["sex"] == "Male")
        ].iloc[0]
        row_5yr = result_5yr[
            (result_5yr["county_fips"] == "38001")
            & (result_5yr["age_group"] == "5-9")
            & (result_5yr["sex"] == "Male")
        ].iloc[0]

        # surv^(4/5) > surv^1 since surv < 1 -> expected_4yr > expected_5yr
        # Actually surv^(4/5) > surv^(5/5) when surv < 1
        assert row_4yr["expected_pop"] >= row_5yr["expected_pop"]

    def test_birth_cohort_zero(self, pop_start, pop_end, survival_rates):
        """0-4 age group at end has migration_rate = 0."""
        result = compute_residual_migration_rates(
            pop_start, pop_end, survival_rates, period=(2000, 2005)
        )
        birth_rows = result[result["age_group"] == "0-4"]
        assert (birth_rows["migration_rate"] == 0.0).all()
        assert (birth_rows["net_migration"] == 0.0).all()


# ---------------------------------------------------------------------------
# TestPeriodDampening
# ---------------------------------------------------------------------------


class TestPeriodDampening:
    """Tests for apply_period_dampening."""

    @pytest.fixture
    def rate_df(self):
        """Synthetic rates for oil and non-oil counties."""
        records = [
            {
                "county_fips": county,
                "age_group": ag,
                "sex": sex,
                "migration_rate": 0.10,
                "net_migration": 50.0,
            }
            for county in ["38105", "38017"]  # Williams (oil) and Cass (non-oil)
            for ag in AGE_GROUP_LABELS
            for sex in ["Male", "Female"]
        ]
        return pd.DataFrame(records)

    @pytest.fixture
    def dampening_config(self):
        return {
            "enabled": True,
            "factor": 0.60,
            "counties": ["38105", "38053", "38061", "38025", "38089"],
            "boom_periods": [[2005, 2010], [2010, 2015]],
        }

    def test_boom_period_dampened(self, rate_df, dampening_config):
        """Oil counties in boom period get dampened."""
        result = apply_period_dampening(
            rate_df, period=(2005, 2010), dampening_config=dampening_config
        )

        oil_rows = result[result["county_fips"] == "38105"]
        assert np.allclose(oil_rows["migration_rate"], 0.10 * 0.60)
        assert np.allclose(oil_rows["net_migration"], 50.0 * 0.60)

    def test_non_boom_period_unchanged(self, rate_df, dampening_config):
        """All counties unchanged in non-boom periods."""
        result = apply_period_dampening(
            rate_df, period=(2015, 2020), dampening_config=dampening_config
        )

        assert np.allclose(result["migration_rate"], 0.10)
        assert np.allclose(result["net_migration"], 50.0)

    def test_non_oil_county_unchanged(self, rate_df, dampening_config):
        """Non-oil counties unchanged even in boom periods."""
        result = apply_period_dampening(
            rate_df, period=(2005, 2010), dampening_config=dampening_config
        )

        cass_rows = result[result["county_fips"] == "38017"]
        assert np.allclose(cass_rows["migration_rate"], 0.10)
        assert np.allclose(cass_rows["net_migration"], 50.0)

    def test_dampening_disabled(self, rate_df):
        """Disabled dampening returns unchanged data."""
        config = {"enabled": False}
        result = apply_period_dampening(rate_df, period=(2005, 2010), dampening_config=config)
        assert np.allclose(result["migration_rate"], 0.10)


# ---------------------------------------------------------------------------
# TestCollegeAgeAdjustment
# ---------------------------------------------------------------------------


class TestCollegeAgeAdjustment:
    """Tests for apply_college_age_adjustment."""

    @pytest.fixture
    def rates_with_college(self):
        """Rates with extreme college-age values in college counties."""
        records = []
        for county in ["38035", "38001"]:  # Grand Forks (college) and Adams
            for ag in AGE_GROUP_LABELS:
                for sex in ["Male", "Female"]:
                    if county == "38035" and ag in ("15-19", "20-24"):
                        rate = 0.50  # Very high in-migration
                    else:
                        rate = 0.05
                    records.append(
                        {
                            "county_fips": county,
                            "age_group": ag,
                            "sex": sex,
                            "migration_rate": rate,
                        }
                    )
        return pd.DataFrame(records)

    def test_smooth_reduces_extreme(self, rates_with_college):
        """Smoothing blends college county rates toward statewide average."""
        result = apply_college_age_adjustment(
            rates_with_college,
            college_counties=["38035"],
            method="smooth",
            blend_factor=0.5,
        )

        # College county college-age rates should be reduced
        gf_college = result[
            (result["county_fips"] == "38035") & (result["age_group"].isin(["15-19", "20-24"]))
        ]
        # Should be between 0.05 (state avg) and 0.50 (original)
        assert (gf_college["migration_rate"] < 0.50).all()
        assert (gf_college["migration_rate"] > 0.05).all()

    def test_non_college_county_unchanged(self, rates_with_college):
        """Non-college counties are not affected."""
        result = apply_college_age_adjustment(
            rates_with_college,
            college_counties=["38035"],
            method="smooth",
            blend_factor=0.5,
        )

        adams = result[result["county_fips"] == "38001"]
        assert np.allclose(adams["migration_rate"], 0.05)

    def test_non_college_age_unchanged(self, rates_with_college):
        """Non-college age groups in college counties unchanged."""
        result = apply_college_age_adjustment(
            rates_with_college,
            college_counties=["38035"],
            method="smooth",
            blend_factor=0.5,
        )

        gf_other = result[
            (result["county_fips"] == "38035") & (~result["age_group"].isin(["15-19", "20-24"]))
        ]
        assert np.allclose(gf_other["migration_rate"], 0.05)

    def test_cap_method(self, rates_with_college):
        """Cap method clips extreme rates."""
        result = apply_college_age_adjustment(
            rates_with_college,
            college_counties=["38035"],
            method="cap",
        )

        gf_college = result[
            (result["county_fips"] == "38035") & (result["age_group"].isin(["15-19", "20-24"]))
        ]
        assert (gf_college["migration_rate"] <= 0.20).all()


# ---------------------------------------------------------------------------
# TestMaleMigrationDampening
# ---------------------------------------------------------------------------


class TestMaleMigrationDampening:
    """Tests for apply_male_migration_dampening."""

    @pytest.fixture
    def mixed_rates(self):
        """Rates with male and female entries."""
        records = [
            {
                "county_fips": "38001",
                "age_group": ag,
                "sex": sex,
                "migration_rate": 0.10,
                "net_migration": 50.0,
            }
            for ag in AGE_GROUP_LABELS
            for sex in ["Male", "Female"]
        ]
        return pd.DataFrame(records)

    def test_boom_period_dampens_males(self, mixed_rates):
        """Male rates reduced in boom period."""
        result = apply_male_migration_dampening(
            mixed_rates,
            period=(2005, 2010),
            boom_periods=[[2005, 2010], [2010, 2015]],
            male_dampening_factor=0.80,
        )

        males = result[result["sex"] == "Male"]
        assert np.allclose(males["migration_rate"], 0.10 * 0.80)

        females = result[result["sex"] == "Female"]
        assert np.allclose(females["migration_rate"], 0.10)

    def test_non_boom_unchanged(self, mixed_rates):
        """Non-boom periods leave all rates unchanged."""
        result = apply_male_migration_dampening(
            mixed_rates,
            period=(2015, 2020),
            boom_periods=[[2005, 2010], [2010, 2015]],
            male_dampening_factor=0.80,
        )

        assert np.allclose(result["migration_rate"], 0.10)


# ---------------------------------------------------------------------------
# TestMultiPeriodAveraging
# ---------------------------------------------------------------------------


class TestMultiPeriodAveraging:
    """Tests for average_period_rates."""

    def test_simple_average(self):
        """Simple average of two periods."""
        period_1 = pd.DataFrame(
            [
                {
                    "county_fips": "38001",
                    "age_group": "0-4",
                    "sex": "Male",
                    "migration_rate": 0.10,
                    "net_migration": 50.0,
                },
            ]
        )
        period_2 = pd.DataFrame(
            [
                {
                    "county_fips": "38001",
                    "age_group": "0-4",
                    "sex": "Male",
                    "migration_rate": 0.20,
                    "net_migration": 100.0,
                },
            ]
        )

        result = average_period_rates(
            {(2000, 2005): period_1, (2005, 2010): period_2},
            method="simple_average",
        )

        assert len(result) == 1
        np.testing.assert_allclose(result["migration_rate"].values[0], 0.15)
        np.testing.assert_allclose(result["net_migration"].values[0], 75.0)
        assert result["n_periods"].values[0] == 2

    def test_empty_input(self):
        """Empty input returns empty DataFrame."""
        result = average_period_rates({})
        assert len(result) == 0
        assert "migration_rate" in result.columns

    def test_single_period(self):
        """Single period returns that period's rates."""
        period = pd.DataFrame(
            [
                {
                    "county_fips": "38001",
                    "age_group": "0-4",
                    "sex": "Male",
                    "migration_rate": 0.15,
                    "net_migration": 75.0,
                },
            ]
        )

        result = average_period_rates({(2000, 2005): period}, method="simple_average")
        assert len(result) == 1
        np.testing.assert_allclose(result["migration_rate"].values[0], 0.15)
        assert result["n_periods"].values[0] == 1

    def test_multi_county_averaging(self):
        """Averaging preserves per-county rates independently."""
        period_1 = pd.DataFrame(
            [
                {
                    "county_fips": "38001",
                    "age_group": "0-4",
                    "sex": "Male",
                    "migration_rate": 0.10,
                    "net_migration": 50.0,
                },
                {
                    "county_fips": "38017",
                    "age_group": "0-4",
                    "sex": "Male",
                    "migration_rate": 0.20,
                    "net_migration": 200.0,
                },
            ]
        )
        period_2 = pd.DataFrame(
            [
                {
                    "county_fips": "38001",
                    "age_group": "0-4",
                    "sex": "Male",
                    "migration_rate": 0.30,
                    "net_migration": 150.0,
                },
                {
                    "county_fips": "38017",
                    "age_group": "0-4",
                    "sex": "Male",
                    "migration_rate": 0.40,
                    "net_migration": 400.0,
                },
            ]
        )

        result = average_period_rates(
            {(2000, 2005): period_1, (2005, 2010): period_2},
            method="simple_average",
        )

        c1 = result[result["county_fips"] == "38001"].iloc[0]
        c2 = result[result["county_fips"] == "38017"].iloc[0]

        np.testing.assert_allclose(c1["migration_rate"], 0.20)  # (0.10+0.30)/2
        np.testing.assert_allclose(c2["migration_rate"], 0.30)  # (0.20+0.40)/2


# ---------------------------------------------------------------------------
# TestEndToEnd (integration test using actual data files)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration tests using actual data files.

    These tests are skipped if the required data files are not present.
    """

    PROJECT_ROOT = Path(__file__).parent.parent.parent

    @pytest.fixture
    def has_data_files(self):
        """Check whether required data files exist."""
        required = [
            self.PROJECT_ROOT
            / "data"
            / "raw"
            / "nd_sdc_2024_projections"
            / "source_files"
            / "reference"
            / "Census 2000 County Age and Sex.xlsx",
            self.PROJECT_ROOT
            / "data"
            / "processed"
            / "sdc_2024"
            / "survival_rates_sdc_2024_by_age_group.csv",
            self.PROJECT_ROOT / "sdc_2024_replication" / "data" / "base_population_by_county.csv",
        ]
        for p in required:
            if not p.exists():
                pytest.skip(f"Data file not found: {p}")
        return True

    def test_census_2000_loader(self, has_data_files):
        """Load Census 2000 data and verify shape."""
        from cohort_projections.data.load.census_age_sex_population import (
            load_census_2000_county_age_sex,
        )

        path = (
            self.PROJECT_ROOT
            / "data"
            / "raw"
            / "nd_sdc_2024_projections"
            / "source_files"
            / "reference"
            / "Census 2000 County Age and Sex.xlsx"
        )
        result = load_census_2000_county_age_sex(path, state_fips="38", year=2000)

        # 53 counties * 18 age groups * 2 sexes = 1908
        assert len(result) == 1908
        assert result["county_fips"].nunique() == 53
        assert set(result["sex"].unique()) == {"Male", "Female"}
        assert set(result["age_group"].unique()) == set(AGE_GROUP_LABELS)
        assert result["population"].sum() > 600_000  # ND pop ~642k in 2000

    def test_base_population_2020_loader(self, has_data_files):
        """Load 2020 base population and verify shape."""
        from cohort_projections.data.load.census_age_sex_population import (
            load_census_2020_base_population,
        )

        path = self.PROJECT_ROOT / "sdc_2024_replication" / "data" / "base_population_by_county.csv"
        result = load_census_2020_base_population(path)

        assert len(result) == 1908
        assert result["county_fips"].nunique() == 53
        assert result["population"].sum() > 700_000  # ND pop ~779k in 2020

    def test_residual_computation_single_period(self, has_data_files):
        """Compute residual migration for 2000-2005 period."""
        from cohort_projections.data.load.census_age_sex_population import (
            load_census_2000_county_age_sex,
        )

        path = (
            self.PROJECT_ROOT
            / "data"
            / "raw"
            / "nd_sdc_2024_projections"
            / "source_files"
            / "reference"
            / "Census 2000 County Age and Sex.xlsx"
        )

        pop_2000 = load_census_2000_county_age_sex(path, year=2000)
        pop_2005 = load_census_2000_county_age_sex(path, year=2005)

        surv_path = (
            self.PROJECT_ROOT
            / "data"
            / "processed"
            / "sdc_2024"
            / "survival_rates_sdc_2024_by_age_group.csv"
        )
        surv = pd.read_csv(surv_path)

        result = compute_residual_migration_rates(pop_2000, pop_2005, surv, period=(2000, 2005))

        # 53 counties * 18 age groups * 2 sexes
        assert len(result) == 53 * 18 * 2
        assert result["county_fips"].nunique() == 53

        # Net migration should be reasonable (not all zero)
        non_birth = result[result["age_group"] != "0-4"]
        assert non_birth["net_migration"].abs().sum() > 0
