"""
Unit tests for mortality_improvement.py (Phase 3 Census Bureau mortality improvement).

Tests the full pipeline that computes time-varying ND-adjusted survival rates
from Census Bureau NP2023 national projections. Uses synthetic data fixtures
for unit tests and real data files for the end-to-end integration test.
"""

from pathlib import Path

import pandas as pd
import pytest

from cohort_projections.data.process.mortality_improvement import (
    build_nd_adjusted_survival_projections,
    compute_nd_adjustment_factors,
    load_census_survival_projections,
    load_nd_baseline_survival,
    run_mortality_improvement_pipeline,
)

# Project root for locating real data files
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Real data file paths
CENSUS_SURVIVAL_FILE = (
    PROJECT_ROOT / "data" / "raw" / "census_bureau_methodology" / "np2023_a4_survival_ratios.csv"
)
ND_BASELINE_FILE = (
    PROJECT_ROOT / "data" / "processed" / "sdc_2024" / "survival_rates_sdc_2024_by_age_group.csv"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def census_wide_csv(tmp_path: Path) -> Path:
    """Create a synthetic Census NP2023-A4 wide-format CSV."""
    rows = []
    for sex in [1, 2]:
        for year in range(2025, 2046):
            # Generate plausible survival ratios by age
            srat_values = {}
            for age in range(101):
                if age == 0:
                    base = 0.995 if sex == 2 else 0.993
                elif age < 15:
                    base = 0.9999
                elif age < 65:
                    base = 0.999 - (age - 15) * 0.00003
                elif age < 85:
                    base = 0.995 - (age - 65) * 0.002
                else:
                    base = 0.955 - (age - 85) * 0.008

                # Small annual improvement
                improvement = (year - 2025) * 0.00002
                srat_values[f"SRAT_{age}"] = min(max(base + improvement, 0.5), 1.0)

            row = {"NATIVITY": 0, "SEX": sex, "GROUP": 0, "YEAR": year}
            row.update(srat_values)
            rows.append(row)

    df = pd.DataFrame(rows)
    file_path = tmp_path / "np2023_a4_survival_ratios.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def nd_baseline_csv(tmp_path: Path) -> Path:
    """Create a synthetic ND CDC baseline survival rates CSV."""
    age_groups = [
        ("0-4", 0, 4),
        ("5-9", 5, 9),
        ("10-14", 10, 14),
        ("15-19", 15, 19),
        ("20-24", 20, 24),
        ("25-29", 25, 29),
        ("30-34", 30, 34),
        ("35-39", 35, 39),
        ("40-44", 40, 44),
        ("45-49", 45, 49),
        ("50-54", 50, 54),
        ("55-59", 55, 59),
        ("60-64", 60, 64),
        ("65-69", 65, 69),
        ("70-74", 70, 74),
        ("75-79", 75, 79),
        ("80-84", 80, 84),
        ("85+", 85, 89),
    ]
    rows = []
    for sex in ["Female", "Male"]:
        for group_name, age_start, age_end in age_groups:
            # Generate plausible 1-year survival rates
            mid_age = (age_start + age_end) / 2
            if mid_age < 5:
                rate_1yr = 0.999 if sex == "Female" else 0.998
            elif mid_age < 15:
                rate_1yr = 0.99988
            elif mid_age < 50:
                rate_1yr = 0.9995 - (mid_age - 15) * 0.00002
            elif mid_age < 65:
                rate_1yr = 0.997 - (mid_age - 50) * 0.0003
            elif mid_age < 80:
                rate_1yr = 0.992 - (mid_age - 65) * 0.003
            else:
                rate_1yr = 0.960 - (mid_age - 80) * 0.005

            # Small sex differential
            if sex == "Female":
                rate_1yr = min(rate_1yr * 1.002, 1.0)

            # Compute 5-year rate for completeness (not used by our module)
            rate_5yr = rate_1yr**5

            rows.append(
                {
                    "age_group": group_name,
                    "age_start": age_start,
                    "age_end": age_end,
                    "sex": sex,
                    "survival_rate_5yr": round(rate_5yr, 6),
                    "survival_rate_1yr": rate_1yr,
                    "source": "test_synthetic",
                    "notes": "Synthetic test data",
                }
            )

    df = pd.DataFrame(rows)
    file_path = tmp_path / "survival_rates_sdc_2024_by_age_group.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def census_long_df(census_wide_csv: Path) -> pd.DataFrame:
    """Load synthetic census data in long format."""
    return load_census_survival_projections(census_wide_csv, years=(2025, 2045))


@pytest.fixture
def nd_baseline_df(nd_baseline_csv: Path) -> pd.DataFrame:
    """Load synthetic ND baseline data expanded to single-year ages."""
    return load_nd_baseline_survival(nd_baseline_csv)


@pytest.fixture
def adjustment_factors_df(
    nd_baseline_df: pd.DataFrame,
    census_long_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute adjustment factors from synthetic data."""
    census_2025 = census_long_df[census_long_df["year"] == 2025].copy()
    return compute_nd_adjustment_factors(nd_baseline_df, census_2025)


# ---------------------------------------------------------------------------
# TestLoadCensusSurvivalProjections
# ---------------------------------------------------------------------------


class TestLoadCensusSurvivalProjections:
    """Tests for load_census_survival_projections."""

    def test_loads_correct_year_range(self, census_wide_csv: Path) -> None:
        """Census data is filtered to the requested year range."""
        df = load_census_survival_projections(census_wide_csv, years=(2030, 2035))
        assert df["year"].min() == 2030
        assert df["year"].max() == 2035
        assert sorted(df["year"].unique()) == list(range(2030, 2036))

    def test_filters_sex_correctly(self, census_wide_csv: Path) -> None:
        """Only requested sex codes appear in the output."""
        # Male only
        df_male = load_census_survival_projections(
            census_wide_csv, years=(2025, 2025), sex_filter=[1]
        )
        assert set(df_male["sex"].unique()) == {"Male"}

        # Female only
        df_female = load_census_survival_projections(
            census_wide_csv, years=(2025, 2025), sex_filter=[2]
        )
        assert set(df_female["sex"].unique()) == {"Female"}

    def test_long_format_output(self, census_wide_csv: Path) -> None:
        """Output is in long format with expected columns."""
        df = load_census_survival_projections(census_wide_csv, years=(2025, 2025))
        assert list(df.columns) == ["year", "age", "sex", "survival_ratio"]

    def test_survival_ratios_in_valid_range(self, census_wide_csv: Path) -> None:
        """All survival ratios are in the range (0, 1]."""
        df = load_census_survival_projections(census_wide_csv, years=(2025, 2045))
        assert (df["survival_ratio"] > 0).all()
        assert (df["survival_ratio"] <= 1).all()

    def test_correct_number_of_ages(self, census_wide_csv: Path) -> None:
        """Ages 0-100 = 101 unique ages per sex per year."""
        df = load_census_survival_projections(census_wide_csv, years=(2025, 2025))
        assert df["age"].nunique() == 101
        assert df["age"].min() == 0
        assert df["age"].max() == 100

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_census_survival_projections(tmp_path / "nonexistent.csv", years=(2025, 2045))

    def test_empty_after_filter_raises(self, census_wide_csv: Path) -> None:
        """ValueError when no data remains after filtering."""
        with pytest.raises(ValueError, match="No data after filtering"):
            load_census_survival_projections(census_wide_csv, years=(1900, 1905))


# ---------------------------------------------------------------------------
# TestLoadNDBaseline
# ---------------------------------------------------------------------------


class TestLoadNDBaseline:
    """Tests for load_nd_baseline_survival."""

    def test_expands_to_single_year_ages(self, nd_baseline_csv: Path) -> None:
        """All ages 0-100 present after expansion."""
        df = load_nd_baseline_survival(nd_baseline_csv)
        assert set(df["age"].unique()) == set(range(101))

    def test_85plus_expands_to_ages_85_through_100(self, nd_baseline_csv: Path) -> None:
        """The 85+ group expands to ages 85, 86, ..., 100."""
        df = load_nd_baseline_survival(nd_baseline_csv)
        ages_85_plus = df[df["age"] >= 85]["age"].unique()
        assert set(ages_85_plus) == set(range(85, 101))

    def test_both_sexes_present(self, nd_baseline_csv: Path) -> None:
        """Both Male and Female are present."""
        df = load_nd_baseline_survival(nd_baseline_csv)
        assert set(df["sex"].unique()) == {"Male", "Female"}

    def test_total_row_count(self, nd_baseline_csv: Path) -> None:
        """101 ages x 2 sexes = 202 rows."""
        df = load_nd_baseline_survival(nd_baseline_csv)
        assert len(df) == 101 * 2

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_nd_baseline_survival(tmp_path / "nonexistent.csv")

    def test_rates_within_group_are_same(self, nd_baseline_csv: Path) -> None:
        """All ages within a 5-year group share the same survival rate."""
        df = load_nd_baseline_survival(nd_baseline_csv)
        # Check ages 10, 11, 12, 13, 14 for Male
        group_10_14 = df[(df["age"].between(10, 14)) & (df["sex"] == "Male")]
        assert group_10_14["survival_rate_1yr"].nunique() == 1


# ---------------------------------------------------------------------------
# TestNDAdjustmentFactors
# ---------------------------------------------------------------------------


class TestNDAdjustmentFactors:
    """Tests for compute_nd_adjustment_factors."""

    def test_adjustment_factor_computation(self, adjustment_factors_df: pd.DataFrame) -> None:
        """Adjustment factors are computed for all age-sex combinations."""
        assert len(adjustment_factors_df) == 101 * 2
        assert "adjustment_factor" in adjustment_factors_df.columns

    def test_nd_higher_mortality_produces_factor_below_1(self) -> None:
        """When ND has lower survival than national, factor < 1.0."""
        nd = pd.DataFrame({"age": [50], "sex": ["Male"], "survival_rate_1yr": [0.990]})
        census = pd.DataFrame({"age": [50], "sex": ["Male"], "survival_ratio": [0.995]})
        result = compute_nd_adjustment_factors(nd, census)
        assert result["adjustment_factor"].iloc[0] < 1.0

    def test_nd_lower_mortality_produces_factor_above_1(self) -> None:
        """When ND has higher survival than national, factor > 1.0."""
        nd = pd.DataFrame({"age": [25], "sex": ["Female"], "survival_rate_1yr": [0.9998]})
        census = pd.DataFrame({"age": [25], "sex": ["Female"], "survival_ratio": [0.9990]})
        result = compute_nd_adjustment_factors(nd, census)
        assert result["adjustment_factor"].iloc[0] > 1.0

    def test_all_ages_and_sexes_present(self, adjustment_factors_df: pd.DataFrame) -> None:
        """Factors cover ages 0-100 for both sexes."""
        assert set(adjustment_factors_df["age"].unique()) == set(range(101))
        assert set(adjustment_factors_df["sex"].unique()) == {"Male", "Female"}

    def test_factors_are_reasonable(self, adjustment_factors_df: pd.DataFrame) -> None:
        """Adjustment factors are close to 1.0 (not wildly off)."""
        factors = adjustment_factors_df["adjustment_factor"]
        # Should be within a reasonable range for ND vs national
        assert factors.min() > 0.5
        assert factors.max() < 2.0
        # Mean should be near 1.0
        assert abs(factors.mean() - 1.0) < 0.1


# ---------------------------------------------------------------------------
# TestBuildNDAdjustedProjections
# ---------------------------------------------------------------------------


class TestBuildNDAdjustedProjections:
    """Tests for build_nd_adjusted_survival_projections."""

    def test_year_indexed_output(
        self,
        census_long_df: pd.DataFrame,
        adjustment_factors_df: pd.DataFrame,
    ) -> None:
        """Output contains all requested years."""
        result = build_nd_adjusted_survival_projections(
            census_long_df, adjustment_factors_df, years=(2025, 2045)
        )
        expected_years = set(range(2025, 2046))
        assert set(result["year"].unique()) == expected_years

    def test_survival_capped_at_1(
        self,
        census_long_df: pd.DataFrame,
    ) -> None:
        """Adjusted survival rates never exceed 1.0."""
        # Create artificially high adjustment factors
        high_factors = pd.DataFrame(
            {
                "age": list(range(101)) * 2,
                "sex": ["Male"] * 101 + ["Female"] * 101,
                "adjustment_factor": [1.5] * 202,
            }
        )
        result = build_nd_adjusted_survival_projections(
            census_long_df, high_factors, years=(2025, 2025)
        )
        assert (result["survival_rate"] <= 1.0).all()

    def test_survival_floored_at_0(
        self,
        census_long_df: pd.DataFrame,
    ) -> None:
        """Adjusted survival rates never go below 0.0."""
        # Create near-zero adjustment factors
        low_factors = pd.DataFrame(
            {
                "age": list(range(101)) * 2,
                "sex": ["Male"] * 101 + ["Female"] * 101,
                "adjustment_factor": [0.001] * 202,
            }
        )
        result = build_nd_adjusted_survival_projections(
            census_long_df, low_factors, years=(2025, 2025)
        )
        assert (result["survival_rate"] >= 0.0).all()

    def test_improvement_over_time(
        self,
        census_long_df: pd.DataFrame,
        adjustment_factors_df: pd.DataFrame,
    ) -> None:
        """Mean survival rate generally increases from 2025 to 2045."""
        result = build_nd_adjusted_survival_projections(
            census_long_df, adjustment_factors_df, years=(2025, 2045)
        )
        mean_by_year = result.groupby("year")["survival_rate"].mean()
        # The 2045 mean should be >= the 2025 mean (mortality improvement)
        assert mean_by_year.loc[2045] >= mean_by_year.loc[2025]

    def test_male_female_differential_preserved(
        self,
        census_long_df: pd.DataFrame,
        adjustment_factors_df: pd.DataFrame,
    ) -> None:
        """Female survival remains higher than male (on average)."""
        result = build_nd_adjusted_survival_projections(
            census_long_df, adjustment_factors_df, years=(2025, 2025)
        )
        mean_by_sex = result.groupby("sex")["survival_rate"].mean()
        # Females typically have higher survival
        assert mean_by_sex["Female"] >= mean_by_sex["Male"]

    def test_output_has_source_column(
        self,
        census_long_df: pd.DataFrame,
        adjustment_factors_df: pd.DataFrame,
    ) -> None:
        """Output has source column with expected value."""
        result = build_nd_adjusted_survival_projections(
            census_long_df, adjustment_factors_df, years=(2025, 2025)
        )
        assert "source" in result.columns
        assert (result["source"] == "Census_NP2023_ND_adjusted").all()

    def test_output_row_count(
        self,
        census_long_df: pd.DataFrame,
        adjustment_factors_df: pd.DataFrame,
    ) -> None:
        """Output has expected number of rows: years x ages x sexes."""
        result = build_nd_adjusted_survival_projections(
            census_long_df, adjustment_factors_df, years=(2025, 2045)
        )
        # 21 years x 101 ages x 2 sexes = 4242
        assert len(result) == 21 * 101 * 2


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration tests using real data files."""

    @pytest.mark.skipif(
        not CENSUS_SURVIVAL_FILE.exists() or not ND_BASELINE_FILE.exists(),
        reason="Real data files not available",
    )
    def test_full_pipeline_produces_valid_output(self, tmp_path: Path) -> None:
        """Full pipeline with real data produces valid output."""
        config = {
            "project": {
                "base_year": 2025,
                "projection_horizon": 20,
            },
            "output": {
                "compression": "gzip",
            },
        }

        result = run_mortality_improvement_pipeline(config)

        # Correct total rows: 21 years x 101 ages x 2 sexes = 4242
        assert len(result) == 4242

        # All survival rates in valid range
        assert (result["survival_rate"] > 0).all()
        assert (result["survival_rate"] <= 1.0).all()

        # Correct columns
        assert list(result.columns) == [
            "year",
            "age",
            "sex",
            "survival_rate",
            "source",
        ]

        # Correct year range
        assert result["year"].min() == 2025
        assert result["year"].max() == 2045
        assert result["year"].nunique() == 21

        # Correct age range
        assert result["age"].min() == 0
        assert result["age"].max() == 100
        assert result["age"].nunique() == 101

        # Both sexes
        assert set(result["sex"].unique()) == {"Male", "Female"}

        # Source label
        assert (result["source"] == "Census_NP2023_ND_adjusted").all()

        # Survival improvement over time (mean rate should increase)
        mean_2025 = result[result["year"] == 2025]["survival_rate"].mean()
        mean_2045 = result[result["year"] == 2045]["survival_rate"].mean()
        assert mean_2045 >= mean_2025

    @pytest.mark.skipif(
        not CENSUS_SURVIVAL_FILE.exists() or not ND_BASELINE_FILE.exists(),
        reason="Real data files not available",
    )
    def test_adjustment_factors_are_reasonable_with_real_data(self) -> None:
        """Adjustment factors from real data are near 1.0."""
        census = load_census_survival_projections(CENSUS_SURVIVAL_FILE, years=(2025, 2025))
        nd_baseline = load_nd_baseline_survival(ND_BASELINE_FILE)
        census_2025 = census[census["year"] == 2025].copy()
        factors = compute_nd_adjustment_factors(nd_baseline, census_2025)

        # Factors should be mostly near 1.0
        mean_factor = factors["adjustment_factor"].mean()
        assert 0.8 < mean_factor < 1.2

        # No wildly extreme values
        assert factors["adjustment_factor"].min() > 0.3
        assert factors["adjustment_factor"].max() < 3.0

    @pytest.mark.skipif(
        not CENSUS_SURVIVAL_FILE.exists(),
        reason="Census survival file not available",
    )
    def test_real_census_file_loads_correctly(self) -> None:
        """Real Census NP2023-A4 file loads and parses correctly."""
        df = load_census_survival_projections(CENSUS_SURVIVAL_FILE, years=(2025, 2045))

        # Should have 21 years x 101 ages x 2 sexes = 4242
        assert len(df) == 4242

        # All survival ratios valid
        assert (df["survival_ratio"] > 0).all()
        assert (df["survival_ratio"] <= 1.0).all()

    @pytest.mark.skipif(
        not ND_BASELINE_FILE.exists(),
        reason="ND baseline file not available",
    )
    def test_real_nd_baseline_loads_correctly(self) -> None:
        """Real ND CDC baseline file loads and expands correctly."""
        df = load_nd_baseline_survival(ND_BASELINE_FILE)

        # 101 ages x 2 sexes = 202
        assert len(df) == 202

        # All ages present
        assert set(df["age"].unique()) == set(range(101))

        # Both sexes
        assert set(df["sex"].unique()) == {"Male", "Female"}

        # Rates in valid range
        assert (df["survival_rate_1yr"] > 0).all()
        assert (df["survival_rate_1yr"] <= 1.0).all()
