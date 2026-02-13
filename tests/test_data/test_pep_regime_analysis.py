"""
Unit tests for pep_regime_analysis.py module.

Tests county classification, regime average computation, weighted average
calculation, PEP data loading, and report generation.  Uses synthetic
DataFrames as fixtures -- does not depend on actual data files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cohort_projections.data.process.pep_regime_analysis import (
    DEFAULT_DAMPENING,
    DEFAULT_REGIME_WEIGHTS,
    METRO_COUNTIES,
    MIGRATION_REGIMES,
    OIL_COUNTIES,
    calculate_regime_averages,
    calculate_regime_weighted_average,
    classify_counties,
    generate_regime_analysis_report,
    load_pep_preferred_estimates,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Representative test counties: 2 oil, 1 metro, 1 rural, 1 overlap-capable
TEST_COUNTIES = ["38105", "38053", "38017", "38001", "38003"]


@pytest.fixture
def all_nd_county_fips() -> list[str]:
    """All 53 ND county FIPS codes (38001 through 38105, odd numbers)."""
    return [f"38{str(i).zfill(3)}" for i in range(1, 106, 2)]


@pytest.fixture
def synthetic_pep_data() -> pd.DataFrame:
    """Synthetic PEP preferred-estimate data for 5 test counties, 2000-2024.

    Generates realistic-looking net migration values:
    - Oil counties: large positive during boom, negative pre-Bakken
    - Metro counties: moderate positive throughout
    - Rural counties: negative throughout (population loss)
    """
    np.random.seed(42)

    records = []
    for fips in TEST_COUNTIES:
        for year in range(2000, 2025):
            if fips in OIL_COUNTIES:
                if year <= 2010:
                    base = -30
                elif year <= 2015:
                    base = 800
                elif year <= 2021:
                    base = 50
                else:
                    base = 200
            elif fips in METRO_COUNTIES:
                base = 250
            else:
                base = -40

            netmig = base + np.random.normal(0, abs(base) * 0.15 + 5)
            records.append(
                {
                    "county_fips": fips,
                    "year": year,
                    "netmig": netmig,
                    "is_preferred_estimate": True,
                    "county_name": f"County {fips}",
                }
            )

    return pd.DataFrame(records)


@pytest.fixture
def synthetic_regime_averages(synthetic_pep_data) -> pd.DataFrame:
    """Regime averages derived from the synthetic PEP fixture."""
    return calculate_regime_averages(synthetic_pep_data)


@pytest.fixture
def synthetic_classifications() -> pd.DataFrame:
    """Classifications for the 5 test counties."""
    return classify_counties(TEST_COUNTIES)


# ---------------------------------------------------------------------------
# TestClassifyCounties
# ---------------------------------------------------------------------------


class TestClassifyCounties:
    """Tests for classify_counties function."""

    def test_all_53_counties_classified(self, all_nd_county_fips):
        """Every county in the input list receives a classification row."""
        result = classify_counties(all_nd_county_fips)
        assert len(result) == len(all_nd_county_fips)

    def test_oil_counties_identified(self):
        """Known oil counties are flagged as is_oil=True."""
        fips_list = list(OIL_COUNTIES.keys())
        result = classify_counties(fips_list)
        assert result["is_oil"].all()
        assert (result["county_type"] == "oil").all()

    def test_metro_counties_identified(self):
        """Known metro counties are flagged as is_metro=True."""
        fips_list = list(METRO_COUNTIES.keys())
        result = classify_counties(fips_list)
        assert result["is_metro"].all()
        assert (result["county_type"] == "metro").all()

    def test_rural_is_complement(self, all_nd_county_fips):
        """Counties not in oil or metro lists are classified as rural."""
        result = classify_counties(all_nd_county_fips)
        oil_metro = set(OIL_COUNTIES.keys()) | set(METRO_COUNTIES.keys())
        rural = result[~result["county_fips"].isin(oil_metro)]
        assert (rural["county_type"] == "rural").all()
        assert not rural["is_oil"].any()
        assert not rural["is_metro"].any()

    def test_custom_classifications(self):
        """Custom oil/metro overrides replace the defaults."""
        custom = {
            "oil": ["38001"],
            "metro": ["38003"],
        }
        result = classify_counties(["38001", "38003", "38005"], custom)

        oil_row = result[result["county_fips"] == "38001"].iloc[0]
        assert bool(oil_row["is_oil"]) is True
        assert oil_row["county_type"] == "oil"

        metro_row = result[result["county_fips"] == "38003"].iloc[0]
        assert bool(metro_row["is_metro"]) is True
        assert metro_row["county_type"] == "metro"

        rural_row = result[result["county_fips"] == "38005"].iloc[0]
        assert rural_row["county_type"] == "rural"

    def test_oil_metro_overlap(self):
        """A county present in both oil and metro lists gets type 'oil_metro'."""
        custom = {
            "oil": ["38017"],  # Cass is normally metro-only
            "metro": ["38017"],
        }
        result = classify_counties(["38017"], custom)
        assert result.iloc[0]["county_type"] == "oil_metro"
        assert bool(result.iloc[0]["is_oil"]) is True
        assert bool(result.iloc[0]["is_metro"]) is True

    def test_output_columns(self):
        """Output DataFrame has the expected columns."""
        result = classify_counties(["38001"])
        expected_cols = {"county_fips", "county_type", "is_oil", "is_metro"}
        assert set(result.columns) == expected_cols


# ---------------------------------------------------------------------------
# TestCalculateRegimeAverages
# ---------------------------------------------------------------------------


class TestCalculateRegimeAverages:
    """Tests for calculate_regime_averages function."""

    def test_correct_period_grouping(self, synthetic_pep_data):
        """Each record's regime label corresponds to the right year range."""
        result = calculate_regime_averages(synthetic_pep_data)

        for _, row in result.iterrows():
            regime_def = MIGRATION_REGIMES[row["regime"]]
            start, end = int(regime_def["start"]), int(regime_def["end"])
            expected_years = end - start + 1

            # n_years should not exceed the regime span
            assert row["n_years"] <= expected_years

    def test_stats_calculated_correctly(self):
        """Mean, median, std, n_years, and total are correct for known data."""
        data = pd.DataFrame(
            {
                "county_fips": ["38001"] * 5,
                "year": [2000, 2001, 2002, 2003, 2004],
                "netmig": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        regimes = {
            "test_regime": {"start": 2000, "end": 2004, "label": "Test"},
        }
        result = calculate_regime_averages(data, regimes=regimes)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["mean_netmig"] == pytest.approx(30.0)
        assert row["median_netmig"] == pytest.approx(30.0)
        assert row["n_years"] == 5
        assert row["total_netmig"] == pytest.approx(150.0)
        assert row["std_netmig"] > 0

    def test_custom_regimes(self, synthetic_pep_data):
        """Custom regime definitions are respected."""
        custom = {
            "early": {"start": 2000, "end": 2005, "label": "Early"},
            "late": {"start": 2020, "end": 2024, "label": "Late"},
        }
        result = calculate_regime_averages(synthetic_pep_data, regimes=custom)
        assert set(result["regime"].unique()) == {"early", "late"}

    def test_handles_missing_years_gracefully(self):
        """Counties with fewer observations than the regime span still work."""
        data = pd.DataFrame(
            {
                "county_fips": ["38001", "38001", "38003"],
                "year": [2000, 2002, 2001],
                "netmig": [10.0, 30.0, 100.0],
            }
        )
        regimes = {
            "period": {"start": 2000, "end": 2004, "label": "Period"},
        }
        result = calculate_regime_averages(data, regimes=regimes)

        county_38001 = result[result["county_fips"] == "38001"]
        assert county_38001.iloc[0]["n_years"] == 2  # Only 2 of 5 years present

        county_38003 = result[result["county_fips"] == "38003"]
        assert county_38003.iloc[0]["n_years"] == 1

    def test_output_columns(self, synthetic_pep_data):
        """Output has the expected column set."""
        result = calculate_regime_averages(synthetic_pep_data)
        expected = {
            "county_fips",
            "regime",
            "mean_netmig",
            "median_netmig",
            "std_netmig",
            "n_years",
            "total_netmig",
        }
        assert set(result.columns) == expected

    def test_single_year_std_is_zero(self):
        """Standard deviation for a single observation is zero, not NaN."""
        data = pd.DataFrame(
            {
                "county_fips": ["38001"],
                "year": [2023],
                "netmig": [42.0],
            }
        )
        regimes = {"r": {"start": 2023, "end": 2023, "label": "One"}}
        result = calculate_regime_averages(data, regimes=regimes)
        assert result.iloc[0]["std_netmig"] == 0.0


# ---------------------------------------------------------------------------
# TestCalculateRegimeWeightedAverage
# ---------------------------------------------------------------------------


class TestCalculateRegimeWeightedAverage:
    """Tests for calculate_regime_weighted_average function."""

    def test_weights_must_sum_to_one(self, synthetic_regime_averages):
        """A ValueError is raised when weights do not sum to 1.0."""
        bad_weights = {"pre_bakken": 0.5, "boom": 0.5, "bust_covid": 0.5, "recovery": 0.5}
        with pytest.raises(ValueError, match="must sum to 1.0"):
            calculate_regime_weighted_average(synthetic_regime_averages, weights=bad_weights)

    def test_dampening_applied_before_weighting(self):
        """Dampening multiplies the raw mean before it is weighted."""
        regime_avg = pd.DataFrame(
            {
                "county_fips": ["38001", "38001"],
                "regime": ["boom", "recovery"],
                "mean_netmig": [100.0, 200.0],
                "median_netmig": [100.0, 200.0],
                "std_netmig": [10.0, 10.0],
                "n_years": [5, 3],
                "total_netmig": [500.0, 600.0],
            }
        )
        weights = {"boom": 0.5, "recovery": 0.5}
        dampening = {"boom": 0.5}  # halve the boom mean

        result = calculate_regime_weighted_average(regime_avg, weights=weights, dampening=dampening)

        # Expected: (100*0.5)*0.5 + 200*0.5 = 25 + 100 = 125
        expected = 125.0
        assert result.iloc[0]["weighted_avg_netmig"] == pytest.approx(expected)

    def test_default_weights_produce_result(self, synthetic_regime_averages):
        """Default weights and dampening produce a non-empty result."""
        result = calculate_regime_weighted_average(synthetic_regime_averages)
        assert len(result) == synthetic_regime_averages["county_fips"].nunique()
        assert "weighted_avg_netmig" in result.columns

    def test_zero_dampening_means_no_change(self):
        """A dampening factor of 1.0 (no dampening) preserves the raw mean."""
        regime_avg = pd.DataFrame(
            {
                "county_fips": ["38001"],
                "regime": ["boom"],
                "mean_netmig": [500.0],
                "median_netmig": [500.0],
                "std_netmig": [10.0],
                "n_years": [5],
                "total_netmig": [2500.0],
            }
        )
        weights = {"boom": 1.0}
        dampening = {"boom": 1.0}  # No dampening

        result = calculate_regime_weighted_average(regime_avg, weights=weights, dampening=dampening)
        assert result.iloc[0]["weighted_avg_netmig"] == pytest.approx(500.0)

    def test_all_counties_get_weighted_average(self, synthetic_regime_averages):
        """Every county appearing in regime averages gets a weighted average."""
        result = calculate_regime_weighted_average(synthetic_regime_averages)
        input_counties = set(synthetic_regime_averages["county_fips"].unique())
        output_counties = set(result["county_fips"].unique())
        assert input_counties == output_counties

    def test_per_regime_columns_present(self, synthetic_regime_averages):
        """Output contains per-regime mean columns for transparency."""
        result = calculate_regime_weighted_average(synthetic_regime_averages)
        for regime_key in DEFAULT_REGIME_WEIGHTS:
            assert f"{regime_key}_mean" in result.columns

    def test_missing_regime_defaults_to_zero(self):
        """County missing a regime gets 0.0 contribution from that regime."""
        regime_avg = pd.DataFrame(
            {
                "county_fips": ["38001"],
                "regime": ["recovery"],
                "mean_netmig": [100.0],
                "median_netmig": [100.0],
                "std_netmig": [5.0],
                "n_years": [3],
                "total_netmig": [300.0],
            }
        )
        weights = {"recovery": 0.5, "boom": 0.5}
        result = calculate_regime_weighted_average(regime_avg, weights=weights, dampening={})
        # boom is missing -> 0.0*0.5 + 100*0.5 = 50
        assert result.iloc[0]["weighted_avg_netmig"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# TestLoadPepPreferredEstimates
# ---------------------------------------------------------------------------


class TestLoadPepPreferredEstimates:
    """Tests for load_pep_preferred_estimates function."""

    @pytest.fixture
    def pep_parquet(self, tmp_path) -> Path:
        """Write a small parquet file with both preferred and non-preferred rows."""
        data = pd.DataFrame(
            {
                "county_fips": ["38001", "38001", "38003", "38003"],
                "year": [2020, 2020, 2021, 2021],
                "netmig": [10, 20, 30, 40],
                "is_preferred_estimate": [True, False, True, False],
            }
        )
        path = tmp_path / "pep_test.parquet"
        data.to_parquet(path, index=False)
        return path

    def test_only_preferred_returned(self, pep_parquet):
        """Only rows with is_preferred_estimate=True are kept."""
        result = load_pep_preferred_estimates(pep_parquet)
        assert len(result) == 2
        assert result["is_preferred_estimate"].all()

    def test_required_columns_present(self, pep_parquet):
        """Returned DataFrame has the columns we need downstream."""
        result = load_pep_preferred_estimates(pep_parquet)
        assert "county_fips" in result.columns
        assert "year" in result.columns
        assert "netmig" in result.columns

    def test_file_not_found(self, tmp_path):
        """FileNotFoundError raised for nonexistent path."""
        with pytest.raises(FileNotFoundError):
            load_pep_preferred_estimates(tmp_path / "nonexistent.parquet")

    def test_missing_preferred_column(self, tmp_path):
        """ValueError raised when is_preferred_estimate column is absent."""
        data = pd.DataFrame({"county_fips": ["38001"], "year": [2020], "netmig": [10]})
        path = tmp_path / "bad.parquet"
        data.to_parquet(path, index=False)

        with pytest.raises(ValueError, match="is_preferred_estimate"):
            load_pep_preferred_estimates(path)


# ---------------------------------------------------------------------------
# TestGenerateRegimeAnalysisReport
# ---------------------------------------------------------------------------


class TestGenerateRegimeAnalysisReport:
    """Tests for generate_regime_analysis_report function."""

    def test_report_written(
        self,
        tmp_path,
        synthetic_pep_data,
        synthetic_regime_averages,
        synthetic_classifications,
    ):
        """Report file is created at the specified path."""
        weighted = calculate_regime_weighted_average(synthetic_regime_averages)
        output_path = tmp_path / "report.md"

        result = generate_regime_analysis_report(
            pep_data=synthetic_pep_data,
            regime_averages=synthetic_regime_averages,
            classifications=synthetic_classifications,
            weighted_averages=weighted,
            output_path=output_path,
        )

        assert result.exists()
        content = result.read_text()
        assert "Migration Regime Analysis" in content
        assert "County Summary" in content
        assert "Top 10 Highest" in content
        assert "Top 10 Lowest" in content

    def test_report_contains_regime_sections(
        self,
        tmp_path,
        synthetic_pep_data,
        synthetic_regime_averages,
        synthetic_classifications,
    ):
        """Report includes a section for each regime."""
        weighted = calculate_regime_weighted_average(synthetic_regime_averages)
        output_path = tmp_path / "report.md"

        generate_regime_analysis_report(
            pep_data=synthetic_pep_data,
            regime_averages=synthetic_regime_averages,
            classifications=synthetic_classifications,
            weighted_averages=weighted,
            output_path=output_path,
        )

        content = output_path.read_text()
        for regime_def in MIGRATION_REGIMES.values():
            assert str(regime_def["label"]) in content


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------


class TestConstants:
    """Sanity checks on module-level constants."""

    def test_default_weights_sum_to_one(self):
        """DEFAULT_REGIME_WEIGHTS must sum to 1.0."""
        assert abs(sum(DEFAULT_REGIME_WEIGHTS.values()) - 1.0) < 1e-9

    def test_regime_keys_match_weights(self):
        """Weight keys must cover all defined regimes."""
        assert set(DEFAULT_REGIME_WEIGHTS.keys()) == set(MIGRATION_REGIMES.keys())

    def test_dampening_keys_subset_of_regimes(self):
        """Dampening keys should be a subset of defined regime keys."""
        assert set(DEFAULT_DAMPENING.keys()).issubset(set(MIGRATION_REGIMES.keys()))

    def test_oil_counties_are_nd(self):
        """All oil county FIPS begin with 38 (North Dakota)."""
        for fips in OIL_COUNTIES:
            assert fips.startswith("38")

    def test_metro_counties_are_nd(self):
        """All metro county FIPS begin with 38 (North Dakota)."""
        for fips in METRO_COUNTIES:
            assert fips.startswith("38")
