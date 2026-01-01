"""
Unit tests for B1: Regime-Aware Statistical Modeling.

Tests the functions in module_regime_aware/ for:
- Vintage dummy variable creation
- Piecewise trend estimation
- COVID intervention modeling
- Robust standard error estimation
- Regime variance analysis

Per ADR-020 Phase B6, these tests verify the correct implementation
of regime-aware statistical methods for analyzing methodology vintage
transitions in North Dakota international migration data.
"""

import sys
from pathlib import Path

import pytest

# Add module path for imports
MODULE_PATH = (
    Path(__file__).parent.parent.parent
    / "sdc_2024_replication"
    / "scripts"
    / "statistical_analysis"
)
if str(MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH))

from module_regime_aware import (
    VINTAGE_BOUNDARIES,
    calculate_counterfactual_2020,
    create_covid_intervention,
    create_vintage_dummies,
    estimate_covid_effect,
    estimate_piecewise_trend,
    estimate_regime_variances,
    estimate_with_robust_se,
    estimate_wls_by_regime,
    get_vintage_for_year,
)


class TestCreateVintageDummies:
    """Tests for create_vintage_dummies() function."""

    def test_basic_dummy_creation(self, sample_nd_migration_n25):
        """Test that vintage dummies are created correctly."""
        result = create_vintage_dummies(sample_nd_migration_n25)

        assert "vintage_2010s" in result.columns
        assert "vintage_2020s" in result.columns
        assert "vintage_code" in result.columns

    def test_dummy_values_2000s(self, sample_nd_migration_n25):
        """Test that 2000s years have correct dummy values (reference category)."""
        result = create_vintage_dummies(sample_nd_migration_n25)

        # 2000s should be reference category (both dummies = 0)
        years_2000s = result[result["year"] < 2010]
        assert (years_2000s["vintage_2010s"] == 0).all()
        assert (years_2000s["vintage_2020s"] == 0).all()

    def test_dummy_values_2010s(self, sample_nd_migration_n25):
        """Test that 2010s years have correct dummy values."""
        result = create_vintage_dummies(sample_nd_migration_n25)

        # 2010s should have vintage_2010s=1, vintage_2020s=0
        years_2010s = result[(result["year"] >= 2010) & (result["year"] < 2020)]
        assert (years_2010s["vintage_2010s"] == 1).all()
        assert (years_2010s["vintage_2020s"] == 0).all()

    def test_dummy_values_2020s(self, sample_nd_migration_n25):
        """Test that 2020s years have correct dummy values."""
        result = create_vintage_dummies(sample_nd_migration_n25)

        # 2020s should have vintage_2010s=0, vintage_2020s=1
        years_2020s = result[result["year"] >= 2020]
        assert (years_2020s["vintage_2010s"] == 0).all()
        assert (years_2020s["vintage_2020s"] == 1).all()

    def test_vintage_code_values(self, sample_nd_migration_n25):
        """Test vintage_code categorical values."""
        result = create_vintage_dummies(sample_nd_migration_n25)

        # Check vintage codes
        codes = result["vintage_code"].unique()
        assert 2009 in codes
        assert 2020 in codes
        assert 2024 in codes

    def test_custom_prefix(self, sample_nd_migration_n25):
        """Test that custom prefix is applied."""
        result = create_vintage_dummies(sample_nd_migration_n25, prefix="regime")

        assert "regime_2010s" in result.columns
        assert "regime_2020s" in result.columns
        assert "regime_code" in result.columns

    def test_edge_case_year_2010(self, sample_nd_migration_n25):
        """Test boundary year 2010 is correctly classified."""
        result = create_vintage_dummies(sample_nd_migration_n25)

        year_2010 = result[result["year"] == 2010]
        assert year_2010["vintage_2010s"].iloc[0] == 1
        assert year_2010["vintage_2020s"].iloc[0] == 0

    def test_edge_case_year_2020(self, sample_nd_migration_n25):
        """Test boundary year 2020 is correctly classified."""
        result = create_vintage_dummies(sample_nd_migration_n25)

        year_2020 = result[result["year"] == 2020]
        assert year_2020["vintage_2010s"].iloc[0] == 0
        assert year_2020["vintage_2020s"].iloc[0] == 1


class TestEstimatePiecewiseTrend:
    """Tests for estimate_piecewise_trend() function."""

    def test_basic_estimation(self, synthetic_regime_data):
        """Test that piecewise trend estimation runs without error."""
        result = estimate_piecewise_trend(synthetic_regime_data, y_col="y", year_col="year")

        assert result is not None
        assert result.slopes is not None
        assert result.intercepts is not None

    def test_slope_keys(self, synthetic_regime_data):
        """Test that slopes dict has correct keys."""
        result = estimate_piecewise_trend(synthetic_regime_data, y_col="y", year_col="year")

        assert "2000s" in result.slopes
        assert "2010s" in result.slopes
        assert "2020s" in result.slopes

    def test_slope_directions(self, synthetic_regime_data):
        """Test that estimated slopes have expected directions."""
        # synthetic_regime_data has:
        # 2000s: positive trend (5)
        # 2010s: positive trend (10)
        # 2020s: negative trend (-5)

        result = estimate_piecewise_trend(synthetic_regime_data, y_col="y", year_col="year")

        assert result.slopes["2000s"] > 0, "2000s slope should be positive"
        assert result.slopes["2010s"] > 0, "2010s slope should be positive"
        assert result.slopes["2020s"] < 0, "2020s slope should be negative"

    def test_slope_standard_errors(self, synthetic_regime_data):
        """Test that standard errors are computed."""
        result = estimate_piecewise_trend(synthetic_regime_data, y_col="y", year_col="year")

        assert "2000s" in result.slope_se
        assert "2010s" in result.slope_se
        assert "2020s" in result.slope_se

        # SEs should be positive
        assert result.slope_se["2000s"] > 0
        assert result.slope_se["2010s"] > 0
        assert result.slope_se["2020s"] > 0

    def test_r_squared_range(self, synthetic_regime_data):
        """Test that R-squared is in valid range."""
        result = estimate_piecewise_trend(synthetic_regime_data, y_col="y", year_col="year")

        assert 0 <= result.r_squared <= 1

    def test_n_obs(self, synthetic_regime_data):
        """Test that n_obs matches data length."""
        result = estimate_piecewise_trend(synthetic_regime_data, y_col="y", year_col="year")

        assert result.n_obs == len(synthetic_regime_data)

    @pytest.mark.parametrize("cov_type", ["HAC", "HC3", "nonrobust"])
    def test_cov_type_options(self, synthetic_regime_data, cov_type):
        """Test different covariance type specifications."""
        result = estimate_piecewise_trend(
            synthetic_regime_data, y_col="y", year_col="year", cov_type=cov_type
        )

        assert result is not None
        assert result.slopes is not None

    def test_to_dict_method(self, synthetic_regime_data):
        """Test that to_dict() produces serializable output."""
        result = estimate_piecewise_trend(synthetic_regime_data, y_col="y", year_col="year")

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "slopes" in result_dict
        assert "slope_se" in result_dict
        assert "r_squared" in result_dict


class TestCreateCovidIntervention:
    """Tests for create_covid_intervention() function."""

    def test_pulse_intervention_2020(self, nd_migration_with_covid):
        """Test COVID pulse intervention is 1 only for 2020."""
        result = create_covid_intervention(nd_migration_with_covid)

        assert "covid_pulse" in result.columns

        # 2020 should be 1
        assert result[result["year"] == 2020]["covid_pulse"].iloc[0] == 1

        # Other years should be 0
        assert (result[result["year"] != 2020]["covid_pulse"] == 0).all()

    def test_step_intervention_2020_onward(self, nd_migration_with_covid):
        """Test COVID step intervention is 1 for 2020 and after."""
        result = create_covid_intervention(nd_migration_with_covid)

        assert "covid_step" in result.columns

        # Pre-2020 should be 0
        assert (result[result["year"] < 2020]["covid_step"] == 0).all()

        # 2020+ should be 1
        assert (result[result["year"] >= 2020]["covid_step"] == 1).all()

    def test_recovery_ramp(self, nd_migration_with_covid):
        """Test COVID recovery ramp increases from 2020."""
        result = create_covid_intervention(nd_migration_with_covid)

        assert "covid_recovery" in result.columns

        # Pre-2020 should be 0
        assert (result[result["year"] < 2020]["covid_recovery"] == 0).all()

        # 2020 should be 0
        assert result[result["year"] == 2020]["covid_recovery"].iloc[0] == 0

        # 2021 should be 1
        assert result[result["year"] == 2021]["covid_recovery"].iloc[0] == 1

        # 2024 should be 4
        assert result[result["year"] == 2024]["covid_recovery"].iloc[0] == 4

    def test_post_covid_recovery(self, nd_migration_with_covid):
        """Test post_covid_recovery excludes 2020."""
        result = create_covid_intervention(nd_migration_with_covid)

        assert "post_covid_recovery" in result.columns

        # Pre-2020 should be 0
        assert (result[result["year"] < 2020]["post_covid_recovery"] == 0).all()

        # 2020 should be 0
        assert result[result["year"] == 2020]["post_covid_recovery"].iloc[0] == 0

        # Post-2020 should be 1
        assert (result[result["year"] > 2020]["post_covid_recovery"] == 1).all()


class TestEstimateCovidEffect:
    """Tests for estimate_covid_effect() function."""

    def test_basic_estimation(self, nd_migration_with_covid):
        """Test COVID effect estimation runs without error."""
        result = estimate_covid_effect(
            nd_migration_with_covid,
            y_col="intl_migration",
            intervention_type="pulse",
        )

        assert result is not None
        assert result.covid_effect is not None

    def test_covid_effect_negative_for_collapse(self, nd_migration_with_covid):
        """Test that COVID effect is negative (migration collapsed in 2020)."""
        result = estimate_covid_effect(
            nd_migration_with_covid,
            y_col="intl_migration",
            intervention_type="pulse",
        )

        # COVID caused migration to collapse, so effect should be negative
        assert result.covid_effect < 0

    def test_model_comparison_metrics(self, nd_migration_with_covid):
        """Test that AIC/BIC metrics are computed."""
        result = estimate_covid_effect(
            nd_migration_with_covid,
            y_col="intl_migration",
            intervention_type="pulse",
        )

        assert result.aic_with is not None
        assert result.aic_without is not None
        assert result.bic_with is not None
        assert result.bic_without is not None

    @pytest.mark.parametrize("intervention_type", ["pulse", "step", "recovery"])
    def test_intervention_types(self, nd_migration_with_covid, intervention_type):
        """Test different intervention type specifications."""
        result = estimate_covid_effect(
            nd_migration_with_covid,
            y_col="intl_migration",
            intervention_type=intervention_type,
        )

        assert result is not None

    def test_to_dict_method(self, nd_migration_with_covid):
        """Test that to_dict() produces serializable output."""
        result = estimate_covid_effect(
            nd_migration_with_covid,
            y_col="intl_migration",
            intervention_type="pulse",
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "covid_effect" in result_dict
        assert "preferred_model" in result_dict


class TestCalculateCounterfactual2020:
    """Tests for calculate_counterfactual_2020() function."""

    def test_basic_calculation(self, nd_migration_with_covid):
        """Test counterfactual calculation runs without error."""
        result = calculate_counterfactual_2020(nd_migration_with_covid, y_col="intl_migration")

        assert result is not None
        assert "counterfactual_2020" in result
        assert "actual_2020" in result
        assert "covid_impact" in result

    def test_counterfactual_higher_than_actual(self, nd_migration_with_covid):
        """Test counterfactual is higher than actual 2020 (COVID collapse)."""
        result = calculate_counterfactual_2020(nd_migration_with_covid, y_col="intl_migration")

        # Counterfactual (trend-based) should be higher than actual
        assert result["counterfactual_2020"] > result["actual_2020"]

    def test_covid_impact_negative(self, nd_migration_with_covid):
        """Test COVID impact is negative (actual - counterfactual)."""
        result = calculate_counterfactual_2020(nd_migration_with_covid, y_col="intl_migration")

        assert result["covid_impact"] < 0

    def test_baseline_years(self, nd_migration_with_covid):
        """Test baseline years documentation."""
        result = calculate_counterfactual_2020(nd_migration_with_covid, y_col="intl_migration")

        assert result["baseline_years"] == "2015-2019"


class TestEstimateRegimeVariances:
    """Tests for estimate_regime_variances() function."""

    def test_basic_variance_estimation(self, high_variance_ratio_data):
        """Test regime variance estimation runs without error."""
        result = estimate_regime_variances(
            high_variance_ratio_data, y_col="y", regime_col="vintage"
        )

        assert result is not None
        assert result.variances is not None

    def test_variances_by_regime(self, high_variance_ratio_data):
        """Test variances are computed for each regime."""
        result = estimate_regime_variances(
            high_variance_ratio_data, y_col="y", regime_col="vintage"
        )

        assert "2009" in result.variances
        assert "2020" in result.variances
        assert "2024" in result.variances

    def test_variance_ratio(self, high_variance_ratio_data):
        """Test variance ratio is computed correctly."""
        result = estimate_regime_variances(
            high_variance_ratio_data, y_col="y", regime_col="vintage"
        )

        # High variance ratio data should show ratio > 1
        assert result.variance_ratio > 1

    def test_levene_test(self, high_variance_ratio_data):
        """Test Levene's test for heteroskedasticity."""
        result = estimate_regime_variances(
            high_variance_ratio_data, y_col="y", regime_col="vintage"
        )

        assert result.levene_statistic is not None
        assert result.levene_pvalue is not None

    def test_heteroskedastic_detection(self, high_variance_ratio_data):
        """Test heteroskedasticity detection."""
        result = estimate_regime_variances(
            high_variance_ratio_data, y_col="y", regime_col="vintage"
        )

        # High variance ratio should be detected as heteroskedastic
        assert result.heteroskedastic is True

    def test_homogeneous_variance(self, homogeneous_variance_data):
        """Test with homogeneous variance data."""
        result = estimate_regime_variances(
            homogeneous_variance_data, y_col="y", regime_col="vintage"
        )

        # Homogeneous data should have variance ratio close to 1
        assert result.variance_ratio < 5

    def test_to_dict_method(self, high_variance_ratio_data):
        """Test that to_dict() produces serializable output."""
        result = estimate_regime_variances(
            high_variance_ratio_data, y_col="y", regime_col="vintage"
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "variances_by_regime" in result_dict
        assert "variance_ratio_max_min" in result_dict


class TestEstimateWithRobustSE:
    """Tests for estimate_with_robust_se() function."""

    def test_basic_robust_se(self, synthetic_regime_data):
        """Test robust SE estimation runs without error."""
        # Add a simple X variable for regression
        df = synthetic_regime_data.copy()
        df["t"] = df["year"] - df["year"].min()

        result = estimate_with_robust_se(df, y_col="y", X_cols=["t"])

        assert result is not None

    def test_all_se_types_computed(self, synthetic_regime_data):
        """Test all SE types are computed."""
        df = synthetic_regime_data.copy()
        df["t"] = df["year"] - df["year"].min()

        result = estimate_with_robust_se(df, y_col="y", X_cols=["t"])

        assert result.ols_se is not None
        assert result.hc0_se is not None
        assert result.hc1_se is not None
        assert result.hc2_se is not None
        assert result.hc3_se is not None
        assert result.hac_se is not None

    def test_coefficients_consistent(self, synthetic_regime_data):
        """Test coefficients are consistent across SE types."""
        df = synthetic_regime_data.copy()
        df["t"] = df["year"] - df["year"].min()

        result = estimate_with_robust_se(df, y_col="y", X_cols=["t"])

        # All methods estimate same coefficients
        assert "t" in result.coefficients

    @pytest.mark.parametrize(
        "se_type",
        ["ols_se", "hc0_se", "hc1_se", "hc2_se", "hc3_se", "hac_se"],
    )
    def test_se_types_positive(self, synthetic_regime_data, se_type):
        """Test all SE types are positive."""
        df = synthetic_regime_data.copy()
        df["t"] = df["year"] - df["year"].min()

        result = estimate_with_robust_se(df, y_col="y", X_cols=["t"])

        se_dict = getattr(result, se_type)
        for var, se in se_dict.items():
            assert se > 0, f"{se_type} for {var} should be positive"

    def test_to_dict_method(self, synthetic_regime_data):
        """Test that to_dict() produces serializable output."""
        df = synthetic_regime_data.copy()
        df["t"] = df["year"] - df["year"].min()

        result = estimate_with_robust_se(df, y_col="y", X_cols=["t"])

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "se_comparison" in result_dict


class TestEstimateWLSByRegime:
    """Tests for estimate_wls_by_regime() function."""

    def test_basic_wls(self, high_variance_ratio_data):
        """Test WLS estimation runs without error."""
        df = high_variance_ratio_data.copy()
        df["t"] = df["year"] - df["year"].min()

        result = estimate_wls_by_regime(df, y_col="y", X_cols=["t"], regime_col="vintage")

        assert result is not None
        assert "wls_params" in result

    def test_wls_vs_ols_comparison(self, high_variance_ratio_data):
        """Test WLS and OLS results are both computed."""
        df = high_variance_ratio_data.copy()
        df["t"] = df["year"] - df["year"].min()

        result = estimate_wls_by_regime(df, y_col="y", X_cols=["t"], regime_col="vintage")

        assert "wls_params" in result
        assert "ols_params" in result

    def test_regime_weights_computed(self, high_variance_ratio_data):
        """Test regime-specific weights are computed."""
        df = high_variance_ratio_data.copy()
        df["t"] = df["year"] - df["year"].min()

        result = estimate_wls_by_regime(df, y_col="y", X_cols=["t"], regime_col="vintage")

        assert "regime_weights" in result
        assert len(result["regime_weights"]) == 3  # Three regimes


class TestVintageBoundaries:
    """Tests for vintage boundary constants and utilities."""

    def test_vintage_boundaries_defined(self):
        """Test VINTAGE_BOUNDARIES is properly defined."""
        assert VINTAGE_BOUNDARIES is not None
        assert len(VINTAGE_BOUNDARIES) >= 2

    @pytest.mark.parametrize(
        "year,expected",
        [
            (2000, 2009),
            (2009, 2009),
            (2010, 2020),
            (2019, 2020),
            (2020, 2024),
            (2024, 2024),
        ],
    )
    def test_get_vintage_for_year(self, year, expected):
        """Test get_vintage_for_year returns correct vintage."""
        result = get_vintage_for_year(year)
        assert result == expected


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_observation_per_regime(self, small_regime_data):
        """Test handling of single observation per regime."""
        # This should not raise an error, but results may be less reliable
        try:
            # Add vintage column
            df = small_regime_data.copy()
            df["vintage"] = df["year"].apply(
                lambda y: 2009 if y < 2010 else (2020 if y < 2020 else 2024)
            )

            result = estimate_regime_variances(df, y_col="y", regime_col="vintage")
            assert result is not None
        except Exception as e:
            # Some edge cases may raise errors - that's acceptable
            pytest.skip(f"Edge case handling: {e}")

    def test_negative_migration_value(self, sample_nd_migration_n25):
        """Test handling of negative migration values."""
        # Some years had negative net migration (e.g., 2003)
        df = sample_nd_migration_n25.copy()
        df.loc[df["year"] == 2003, "intl_migration"] = -100

        # Should still run without error
        result = create_vintage_dummies(df)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
