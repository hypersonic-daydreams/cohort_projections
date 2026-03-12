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
    _get_rogers_castro_age_group_weights,
    _rogers_castro_to_age_group_rates,
    apply_college_age_adjustment,
    apply_male_migration_dampening,
    apply_pep_recalibration,
    apply_period_dampening,
    average_period_rates,
    compute_residual_migration_rates,
)
from cohort_projections.utils.sdc_paths import resolve_sdc_replication_root

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

    def test_migration_rate_is_annualized_from_period_rate(self, survival_rates):
        """Migration rates are converted from period rates to annual rates."""
        pop_s = pd.DataFrame(
            [
                {"county_fips": "38001", "age_group": "0-4", "sex": "Male", "population": 100.0},
            ]
        )
        pop_e = pd.DataFrame(
            [
                {"county_fips": "38001", "age_group": "5-9", "sex": "Male", "population": 150.0},
            ]
        )

        result = compute_residual_migration_rates(pop_s, pop_e, survival_rates, period=(2000, 2005))

        row_5_9 = result[(result["age_group"] == "5-9") & (result["sex"] == "Male")].iloc[0]
        period_rate = row_5_9["net_migration"] / row_5_9["expected_pop"]
        expected_annual = (1.0 + period_rate) ** (1.0 / 5.0) - 1.0
        np.testing.assert_allclose(row_5_9["migration_rate"], expected_annual, rtol=1e-6)


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

    def test_smooth_extended_age_groups_25_29(self):
        """College-age smoothing correctly handles extended 25-29 age group (ADR-061)."""
        records = []
        for county in ["38035", "38001"]:  # Grand Forks (college) and Adams
            for ag in AGE_GROUP_LABELS:
                for sex in ["Male", "Female"]:
                    if county == "38035" and ag in ("15-19", "20-24", "25-29"):
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
        rates = pd.DataFrame(records)

        result = apply_college_age_adjustment(
            rates,
            college_counties=["38035"],
            method="smooth",
            age_groups=["15-19", "20-24", "25-29"],
            blend_factor=0.5,
        )

        # All three age groups in college county should be smoothed
        gf_smoothed = result[
            (result["county_fips"] == "38035")
            & (result["age_group"].isin(["15-19", "20-24", "25-29"]))
        ]
        assert (gf_smoothed["migration_rate"] < 0.50).all()
        assert (gf_smoothed["migration_rate"] > 0.05).all()

        # 30-34 should NOT be smoothed
        gf_30 = result[
            (result["county_fips"] == "38035") & (result["age_group"] == "30-34")
        ]
        assert np.allclose(gf_30["migration_rate"], 0.05)


# ---------------------------------------------------------------------------
# TestGQCorrectionFraction (ADR-061)
# ---------------------------------------------------------------------------


class TestGQCorrectionFraction:
    """Tests for fractional GQ correction in subtract_gq_from_populations."""

    @pytest.fixture
    def populations_and_gq(self):
        """Synthetic population and GQ data for fraction testing."""
        from cohort_projections.data.process.residual_migration import (
            subtract_gq_from_populations,
        )
        pop_data = pd.DataFrame(
            {
                "county_fips": ["38017"] * 4,
                "age_group": ["15-19", "20-24", "25-29", "30-34"],
                "sex": ["Male"] * 4,
                "population": [1000.0, 2000.0, 1500.0, 1200.0],
            }
        )
        gq_data = pd.DataFrame(
            {
                "county_fips": ["38017"] * 4,
                "year": [2020] * 4,
                "age_group": ["15-19", "20-24", "25-29", "30-34"],
                "sex": ["Male"] * 4,
                "gq_population": [200.0, 400.0, 100.0, 50.0],
            }
        )
        return {2020: pop_data}, gq_data, subtract_gq_from_populations

    def test_fraction_1_full_subtraction(self, populations_and_gq):
        """fraction=1.0 subtracts 100% of GQ (original behavior)."""
        pops, gq, subtract_fn = populations_and_gq
        result = subtract_fn(pops, gq, fraction=1.0)
        total = result[2020]["population"].sum()
        assert total == pytest.approx(5700.0 - 750.0)

    def test_fraction_0_no_subtraction(self, populations_and_gq):
        """fraction=0.0 subtracts nothing (Phase 1 only)."""
        pops, gq, subtract_fn = populations_and_gq
        result = subtract_fn(pops, gq, fraction=0.0)
        total = result[2020]["population"].sum()
        assert total == pytest.approx(5700.0)

    def test_fraction_half(self, populations_and_gq):
        """fraction=0.5 subtracts half of GQ population."""
        pops, gq, subtract_fn = populations_and_gq
        result = subtract_fn(pops, gq, fraction=0.5)
        total = result[2020]["population"].sum()
        assert total == pytest.approx(5700.0 - 375.0)

    def test_default_fraction_backward_compatible(self, populations_and_gq):
        """Default call (no fraction arg) behaves like fraction=1.0."""
        pops, gq, subtract_fn = populations_and_gq
        result_default = subtract_fn(pops, gq)
        result_explicit = subtract_fn(pops, gq, fraction=1.0)
        assert result_default[2020]["population"].sum() == result_explicit[2020]["population"].sum()


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
    SDC_REPO_ROOT = resolve_sdc_replication_root(project_root=PROJECT_ROOT, must_exist=False)

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
            self.SDC_REPO_ROOT / "data" / "base_population_by_county.csv",
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

        path = self.SDC_REPO_ROOT / "data" / "base_population_by_county.csv"
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


# ---------------------------------------------------------------------------
# TestRogersCastroAgeGroupWeights
# ---------------------------------------------------------------------------


class TestRogersCastroAgeGroupWeights:
    """Tests for _get_rogers_castro_age_group_weights."""

    def test_weights_sum_to_one(self):
        """Rogers-Castro weights across 18 age groups should sum to 1.0."""
        weights = _get_rogers_castro_age_group_weights()
        total = sum(weights.values())
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_all_age_groups_present(self):
        """All 18 standard age groups should have a weight."""
        weights = _get_rogers_castro_age_group_weights()
        for ag in AGE_GROUP_LABELS:
            assert ag in weights, f"Missing age group: {ag}"

    def test_young_adult_peak_is_prominent(self):
        """The 20-24 or 25-29 group should be among the top-3 weights.

        The Rogers-Castro model has a childhood component (ages 0-4) that
        can rival the young-adult peak when aggregated to 5-year groups.
        We verify that the young-adult peak is prominent, not necessarily
        the single highest.
        """
        weights = _get_rogers_castro_age_group_weights()
        sorted_ags = sorted(weights, key=weights.get, reverse=True)
        top_3 = set(sorted_ags[:3])
        assert top_3 & {"20-24", "25-29"}, (
            f"Expected 20-24 or 25-29 in top-3 weights, got {sorted_ags[:3]}"
        )

    def test_all_weights_nonnegative(self):
        """All weights must be non-negative."""
        weights = _get_rogers_castro_age_group_weights()
        for ag, w in weights.items():
            assert w >= 0, f"Negative weight for {ag}: {w}"


# ---------------------------------------------------------------------------
# TestRogersCastroToAgeGroupRates
# ---------------------------------------------------------------------------


class TestRogersCastroToAgeGroupRates:
    """Tests for _rogers_castro_to_age_group_rates."""

    @pytest.fixture
    def expected_pop(self):
        """Synthetic expected population for one county: uniform 500 per cell."""
        records = [
            {"age_group": ag, "sex": sex, "expected_pop": 500.0}
            for ag in AGE_GROUP_LABELS
            for sex in ["Male", "Female"]
        ]
        return pd.DataFrame(records)

    def test_total_migration_preserved(self, expected_pop):
        """Sum of distributed migration counts should equal the input total."""
        total = -1000.0
        result = _rogers_castro_to_age_group_rates(total, expected_pop, sex_ratio=0.5)
        actual_sum = result["rc_migration"].sum()
        np.testing.assert_allclose(actual_sum, total, rtol=1e-6)

    def test_sex_split(self, expected_pop):
        """Male and female migration should respect the sex_ratio."""
        total = 2000.0
        result = _rogers_castro_to_age_group_rates(total, expected_pop, sex_ratio=0.6)
        male_total = result.loc[result["sex"] == "Male", "rc_migration"].sum()
        female_total = result.loc[result["sex"] == "Female", "rc_migration"].sum()
        np.testing.assert_allclose(male_total, 1200.0, rtol=1e-6)
        np.testing.assert_allclose(female_total, 800.0, rtol=1e-6)

    def test_negative_total_produces_negative_rates(self, expected_pop):
        """Negative total migration produces negative rates."""
        result = _rogers_castro_to_age_group_rates(-500.0, expected_pop, sex_ratio=0.5)
        # All rates should be <= 0 (out-migration)
        assert (result["rc_rate"] <= 0).all()

    def test_zero_total_produces_zero(self, expected_pop):
        """Zero total migration produces all-zero rates."""
        result = _rogers_castro_to_age_group_rates(0.0, expected_pop, sex_ratio=0.5)
        assert (result["rc_rate"] == 0.0).all()
        assert (result["rc_migration"] == 0.0).all()

    def test_correct_shape(self, expected_pop):
        """Result should have 36 rows (18 age groups x 2 sexes)."""
        result = _rogers_castro_to_age_group_rates(-500.0, expected_pop, sex_ratio=0.5)
        assert len(result) == 18 * 2


# ---------------------------------------------------------------------------
# TestApplyPepRecalibration
# ---------------------------------------------------------------------------


class TestApplyPepRecalibration:
    """Tests for apply_pep_recalibration."""

    @pytest.fixture
    def pep_data(self):
        """Synthetic PEP data for two counties across 5 years."""
        records = []
        # Benson (38005): net out-migration each year
        for year in range(2001, 2006):
            records.append({"geoid": "38005", "year": year, "netmig": -60})
        # Sioux (38085): sign reversal period (positive PEP)
        for year in range(2001, 2006):
            records.append({"geoid": "38085", "year": year, "netmig": 20})
        # A non-target county
        for year in range(2001, 2006):
            records.append({"geoid": "38017", "year": year, "netmig": 100})
        return pd.DataFrame(records)

    @pytest.fixture
    def period_rates(self):
        """Synthetic residual migration rates for 3 counties."""
        records = []
        for county in ["38005", "38085", "38017"]:
            for ag in AGE_GROUP_LABELS:
                for sex in ["Male", "Female"]:
                    records.append(
                        {
                            "county_fips": county,
                            "age_group": ag,
                            "sex": sex,
                            "period_start": 2000,
                            "period_end": 2005,
                            "pop_start": 200.0,
                            "expected_pop": 190.0,
                            "pop_end": 170.0,
                            "net_migration": -20.0,
                            "migration_rate": -0.02,
                        }
                    )
        return pd.DataFrame(records)

    def test_same_sign_scaling(self, period_rates, pep_data):
        """When PEP and residual have the same sign, rates are scaled by k."""
        result, meta = apply_pep_recalibration(
            period_rates=period_rates,
            period=(2000, 2005),
            pep_data=pep_data,
            counties=["38005"],
            near_zero_threshold=10.0,
        )

        # Benson PEP total = -60 * 5 = -300
        # Benson residual total = -20.0 * 18 * 2 = -720
        # k = -300 / -720 = 0.41667
        expected_k = 300.0 / 720.0

        benson = result[result["county_fips"] == "38005"]
        expected_rate = -0.02 * expected_k
        np.testing.assert_allclose(
            benson["migration_rate"].values[0], expected_rate, rtol=1e-4
        )

        assert meta["38005"]["method"] == "scaled"
        np.testing.assert_allclose(meta["38005"]["k"], expected_k, rtol=1e-3)

    def test_sign_reversal_uses_rogers_castro(self, period_rates, pep_data):
        """When PEP is positive and residual is negative, Rogers-Castro is used."""
        result, meta = apply_pep_recalibration(
            period_rates=period_rates,
            period=(2000, 2005),
            pep_data=pep_data,
            counties=["38085"],
            near_zero_threshold=10.0,
        )

        assert meta["38085"]["method"] == "rogers_castro"
        assert meta["38085"]["reason"] == "sign_reversal"

        # The recalibrated rates for Sioux should reflect PEP total = +100
        # (positive in-migration), so some rates should be positive
        sioux = result[result["county_fips"] == "38085"]
        # At least the peak young-adult ages should have positive rates
        young_adult = sioux[sioux["age_group"].isin(["20-24", "25-29"])]
        assert (young_adult["migration_rate"] > 0).any()

    def test_non_target_county_unchanged(self, period_rates, pep_data):
        """Non-target counties are passed through unchanged."""
        result, meta = apply_pep_recalibration(
            period_rates=period_rates,
            period=(2000, 2005),
            pep_data=pep_data,
            counties=["38005", "38085"],
            near_zero_threshold=10.0,
        )

        cass = result[result["county_fips"] == "38017"]
        original_cass = period_rates[period_rates["county_fips"] == "38017"]

        np.testing.assert_allclose(
            cass["migration_rate"].values,
            original_cass["migration_rate"].values,
        )
        np.testing.assert_allclose(
            cass["net_migration"].values,
            original_cass["net_migration"].values,
        )

    def test_near_zero_residual_uses_rogers_castro(self, pep_data):
        """When residual total is near zero, Rogers-Castro is used."""
        # Create rates with very small net_migration for Benson
        records = []
        for ag in AGE_GROUP_LABELS:
            for sex in ["Male", "Female"]:
                records.append(
                    {
                        "county_fips": "38005",
                        "age_group": ag,
                        "sex": sex,
                        "period_start": 2000,
                        "period_end": 2005,
                        "pop_start": 200.0,
                        "expected_pop": 200.0,
                        "pop_end": 200.0,
                        "net_migration": 0.1,  # Very small: 0.1 * 36 = 3.6 total
                        "migration_rate": 0.0005,
                    }
                )
        near_zero_rates = pd.DataFrame(records)

        result, meta = apply_pep_recalibration(
            period_rates=near_zero_rates,
            period=(2000, 2005),
            pep_data=pep_data,
            counties=["38005"],
            near_zero_threshold=10.0,
        )

        assert meta["38005"]["method"] == "rogers_castro"
        assert meta["38005"]["reason"] == "near_zero"

    def test_metadata_records_pep_and_residual_totals(self, period_rates, pep_data):
        """Metadata should record both PEP and residual totals."""
        _, meta = apply_pep_recalibration(
            period_rates=period_rates,
            period=(2000, 2005),
            pep_data=pep_data,
            counties=["38005"],
            near_zero_threshold=10.0,
        )

        assert "pep_total" in meta["38005"]
        assert "residual_total" in meta["38005"]
        np.testing.assert_allclose(meta["38005"]["pep_total"], -300.0)
        np.testing.assert_allclose(meta["38005"]["residual_total"], -720.0)

    def test_scaling_preserves_rate_proportions(self, period_rates, pep_data):
        """After scaling, relative rate differences across age groups are preserved."""
        # Give different rates to different age groups for Benson
        modified = period_rates.copy()
        benson_mask = modified["county_fips"] == "38005"
        # Assign rates proportional to age group index
        for i, ag in enumerate(AGE_GROUP_LABELS):
            ag_mask = benson_mask & (modified["age_group"] == ag)
            rate = -0.01 * (i + 1)
            modified.loc[ag_mask, "migration_rate"] = rate
            modified.loc[ag_mask, "net_migration"] = rate * 190.0

        result, _ = apply_pep_recalibration(
            period_rates=modified,
            period=(2000, 2005),
            pep_data=pep_data,
            counties=["38005"],
            near_zero_threshold=10.0,
        )

        benson_orig = modified[modified["county_fips"] == "38005"]
        benson_new = result[result["county_fips"] == "38005"]

        # Ratios of rates between age groups should be preserved
        orig_rates = benson_orig.groupby("age_group")["migration_rate"].first()
        new_rates = benson_new.groupby("age_group")["migration_rate"].first()

        # All should be scaled by the same k
        ratios = new_rates / orig_rates
        # Filter out zero rates
        ratios = ratios[orig_rates != 0]
        assert ratios.std() < 1e-6, "Scaling factor should be uniform across age groups"


# ---------------------------------------------------------------------------
# TestCollegeAgeSmoothingPropagation (ADR-049)
# ---------------------------------------------------------------------------


class TestCollegeAgeSmoothingPropagation:
    """Tests verifying ADR-049: college-age smoothing applied to period-level rates.

    ADR-049 fixes a bug where college-age smoothing was applied only to
    averaged rates, but the convergence pipeline reads period-level rates
    from residual_migration_rates.parquet. The fix applies smoothing to
    each period's rates before combining and saving, so the convergence
    pipeline inherits the smoothing automatically.
    """

    @pytest.fixture
    def multi_period_rates_with_college_spike(self):
        """Synthetic period rates for 3 counties across 2 periods.

        College county (38017 = Cass) has extreme 20-24 in-migration rates.
        Non-college counties have moderate rates. This simulates the real
        data pattern where NDSU enrollment inflates Cass 20-24 rates.
        """
        records = []
        for period_start, period_end in [(2000, 2005), (2005, 2010)]:
            for county in ["38001", "38017", "38035"]:
                # 38017 = Cass (NDSU), 38035 = Grand Forks (UND)
                for ag in AGE_GROUP_LABELS:
                    for sex in ["Male", "Female"]:
                        if county in ("38017", "38035") and ag in ("15-19", "20-24"):
                            # Extreme in-migration for college ages in college counties
                            rate = 0.12
                        else:
                            rate = 0.02
                        records.append(
                            {
                                "county_fips": county,
                                "age_group": ag,
                                "sex": sex,
                                "period_start": period_start,
                                "period_end": period_end,
                                "pop_start": 1000.0,
                                "expected_pop": 980.0,
                                "pop_end": 1020.0,
                                "net_migration": rate * 1000,
                                "migration_rate": rate,
                            }
                        )
        return pd.DataFrame(records)

    def test_period_level_smoothing_reduces_college_age_rates(self):
        """College-age smoothing applied per-period reduces extreme rates.

        When smoothing is applied to period-level rates (as ADR-049 requires),
        the college county's 20-24 rates should be blended with the statewide
        average, reducing them from the raw 12% spike.
        """
        # Build per-period rates dict
        period_rates = {}
        for period_start, period_end in [(2000, 2005), (2005, 2010)]:
            records = []
            for county in ["38001", "38017", "38035"]:
                for ag in AGE_GROUP_LABELS:
                    for sex in ["Male", "Female"]:
                        if county in ("38017", "38035") and ag in ("15-19", "20-24"):
                            rate = 0.12
                        else:
                            rate = 0.02
                        records.append(
                            {
                                "county_fips": county,
                                "age_group": ag,
                                "sex": sex,
                                "migration_rate": rate,
                                "net_migration": rate * 1000,
                            }
                        )
            period_rates[(period_start, period_end)] = pd.DataFrame(records)

        # Apply college-age smoothing to each period (ADR-049 approach)
        college_counties = ["38017", "38035"]
        for period_key in period_rates:
            period_rates[period_key] = apply_college_age_adjustment(
                period_rates[period_key],
                college_counties=college_counties,
                method="smooth",
                age_groups=["15-19", "20-24"],
                blend_factor=0.5,
            )

        # Verify that college-age rates are reduced in each period
        for period_key, rates in period_rates.items():
            cass_college = rates[
                (rates["county_fips"] == "38017")
                & (rates["age_group"].isin(["15-19", "20-24"]))
            ]
            # Should be blended: 0.5 * 0.12 + 0.5 * statewide_avg
            # statewide_avg for 20-24 = mean(0.02, 0.12, 0.12) / 3 counties
            # After smoothing, rates should be < 0.12 but > 0.02
            assert (cass_college["migration_rate"] < 0.12).all(), (
                f"Period {period_key}: college-age rates should be reduced by smoothing"
            )
            assert (cass_college["migration_rate"] > 0.02).all(), (
                f"Period {period_key}: college-age rates should still be above baseline"
            )

    def test_non_college_county_unchanged_in_period_smoothing(self):
        """Non-college counties are not affected by period-level smoothing."""
        records = []
        for county in ["38001", "38017"]:
            for ag in AGE_GROUP_LABELS:
                for sex in ["Male", "Female"]:
                    if county == "38017" and ag in ("15-19", "20-24"):
                        rate = 0.12
                    else:
                        rate = 0.02
                    records.append(
                        {
                            "county_fips": county,
                            "age_group": ag,
                            "sex": sex,
                            "migration_rate": rate,
                            "net_migration": rate * 1000,
                        }
                    )
        rates = pd.DataFrame(records)

        result = apply_college_age_adjustment(
            rates,
            college_counties=["38017"],
            method="smooth",
            age_groups=["15-19", "20-24"],
            blend_factor=0.5,
        )

        # Non-college county (38001) should be completely unchanged
        adams_orig = rates[rates["county_fips"] == "38001"]
        adams_result = result[result["county_fips"] == "38001"]
        np.testing.assert_allclose(
            adams_result["migration_rate"].values,
            adams_orig["migration_rate"].values,
        )

    def test_averaged_rates_inherit_period_smoothing(self):
        """Averaging already-smoothed period rates produces smoothed averages.

        ADR-049 applies smoothing to period-level rates, then averages.
        This test verifies that the averaged result reflects the smoothing
        without needing a second smoothing pass.
        """
        # Build period rates with college spikes
        period_rates = {}
        for period_start, period_end in [(2000, 2005), (2005, 2010)]:
            records = []
            for county in ["38001", "38017"]:
                for ag in AGE_GROUP_LABELS:
                    for sex in ["Male", "Female"]:
                        if county == "38017" and ag in ("15-19", "20-24"):
                            rate = 0.12
                        else:
                            rate = 0.02
                        records.append(
                            {
                                "county_fips": county,
                                "age_group": ag,
                                "sex": sex,
                                "migration_rate": rate,
                                "net_migration": rate * 1000,
                            }
                        )
            period_rates[(period_start, period_end)] = pd.DataFrame(records)

        # Apply smoothing to period-level rates first (ADR-049)
        for period_key in period_rates:
            period_rates[period_key] = apply_college_age_adjustment(
                period_rates[period_key],
                college_counties=["38017"],
                method="smooth",
                age_groups=["15-19", "20-24"],
                blend_factor=0.5,
            )

        # Then average
        averaged = average_period_rates(period_rates, method="simple_average")

        # Check that averaged rates for Cass college ages are smoothed
        cass_college_avg = averaged[
            (averaged["county_fips"] == "38017")
            & (averaged["age_group"].isin(["15-19", "20-24"]))
        ]
        # Must be < raw 0.12 (since period-level smoothing reduced them)
        assert (cass_college_avg["migration_rate"] < 0.12).all(), (
            "Averaged rates should reflect period-level smoothing"
        )

    def test_no_double_smoothing(self):
        """Applying smoothing again to averaged rates would over-correct.

        This test demonstrates that if smoothing is applied at both the
        period level AND the averaged level, rates drop much lower than
        a single application. ADR-049 avoids this by only smoothing at
        the period level.
        """
        # Build a simple 2-county setup
        records = []
        for county in ["38001", "38017"]:
            for ag in AGE_GROUP_LABELS:
                for sex in ["Male", "Female"]:
                    if county == "38017" and ag in ("15-19", "20-24"):
                        rate = 0.12
                    else:
                        rate = 0.02
                    records.append(
                        {
                            "county_fips": county,
                            "age_group": ag,
                            "sex": sex,
                            "migration_rate": rate,
                            "net_migration": rate * 1000,
                        }
                    )
        raw_rates = pd.DataFrame(records)

        # Single smoothing (ADR-049 approach)
        single_smoothed = apply_college_age_adjustment(
            raw_rates,
            college_counties=["38017"],
            method="smooth",
            age_groups=["15-19", "20-24"],
            blend_factor=0.5,
        )

        # Double smoothing (the bug scenario: smooth, then smooth again)
        double_smoothed = apply_college_age_adjustment(
            single_smoothed,
            college_counties=["38017"],
            method="smooth",
            age_groups=["15-19", "20-24"],
            blend_factor=0.5,
        )

        # Get rates for Cass college ages
        mask = (raw_rates["county_fips"] == "38017") & (
            raw_rates["age_group"].isin(["15-19", "20-24"])
        )
        single_rates = single_smoothed.loc[mask, "migration_rate"].values
        double_rates = double_smoothed.loc[mask, "migration_rate"].values

        # Double-smoothed should be lower than single-smoothed
        # (over-correction that ADR-049 avoids)
        assert (double_rates < single_rates).all(), (
            "Double smoothing over-corrects: rates drop further than intended"
        )

    def test_smoothed_period_rates_propagate_to_convergence_input(
        self, multi_period_rates_with_college_spike
    ):
        """Period-level smoothing means convergence pipeline gets smoothed rates.

        The convergence pipeline reads from residual_migration_rates.parquet
        which contains the combined period-level rates. This test verifies
        that after applying college-age smoothing per-period and then
        concatenating (simulating the pipeline save), the resulting data
        has smoothed college-age rates.
        """
        all_period_rates = multi_period_rates_with_college_spike

        # Group into per-period dicts for smoothing
        period_groups = {}
        for (ps, pe), group_df in all_period_rates.groupby(
            ["period_start", "period_end"]
        ):
            period_groups[(int(ps), int(pe))] = group_df.reset_index(drop=True)

        # Apply smoothing per-period (as the pipeline does)
        college_counties = ["38017", "38035"]
        for period_key in period_groups:
            period_groups[period_key] = apply_college_age_adjustment(
                period_groups[period_key],
                college_counties=college_counties,
                method="smooth",
                age_groups=["15-19", "20-24"],
                blend_factor=0.5,
            )

        # Combine (simulates what gets saved to residual_migration_rates.parquet)
        combined = pd.concat(list(period_groups.values()), ignore_index=True)

        # Verify: for each period, college county 20-24 rates are < raw 0.12
        for ps, pe in [(2000, 2005), (2005, 2010)]:
            period_data = combined[
                (combined["period_start"] == ps) & (combined["period_end"] == pe)
            ]
            cass_college = period_data[
                (period_data["county_fips"] == "38017")
                & (period_data["age_group"].isin(["20-24"]))
            ]
            assert (cass_college["migration_rate"] < 0.12).all(), (
                f"Period {ps}-{pe}: convergence input should have smoothed "
                f"college-age rates, but found raw values"
            )

        # Verify: non-college county (38001) unchanged across all periods
        adams_rows = combined[combined["county_fips"] == "38001"]
        assert np.allclose(adams_rows["migration_rate"], 0.02), (
            "Non-college county rates should be unchanged after smoothing"
        )
