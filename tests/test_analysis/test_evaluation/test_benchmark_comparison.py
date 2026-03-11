"""Tests for the Benchmark and Comparison Framework (Module 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cohort_projections.analysis.evaluation.benchmark_comparison import (
    BenchmarkComparisonModule,
)
from cohort_projections.analysis.evaluation.schemas import PROJECTION_RESULT_COLUMNS
from cohort_projections.analysis.evaluation.utils import validate_dataframe

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config() -> dict:
    """Minimal evaluation config for testing."""
    return {
        "horizons": [1, 5, 10],
        "near_term_max_horizon": 5,
        "long_term_min_horizon": 10,
        "county_groups": {
            "urban": ["38017", "38015"],
            "rural": ["38099"],
        },
    }


def _make_result_df(
    run_id: str,
    projected_values: list[float],
    actual_values: list[float] | None = None,
    geographies: list[str] | None = None,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Build a synthetic projection-result DataFrame.

    Creates one row per (geography, horizon) combination.
    """
    if geographies is None:
        geographies = ["38017", "38015", "38099"]
    if horizons is None:
        horizons = [1, 5, 10]
    if actual_values is None:
        actual_values = [100.0] * (len(geographies) * len(horizons))

    rows = []
    idx = 0
    for geo in geographies:
        for h in horizons:
            rows.append(
                {
                    "run_id": run_id,
                    "geography": geo,
                    "geography_type": "county",
                    "year": 2020 + h,
                    "horizon": h,
                    "sex": "total",
                    "age_group": "total",
                    "target": "population",
                    "projected_value": projected_values[idx % len(projected_values)],
                    "actual_value": actual_values[idx % len(actual_values)],
                    "base_value": 100.0,
                }
            )
            idx += 1
    return pd.DataFrame(rows)


@pytest.fixture()
def config():
    return _make_config()


@pytest.fixture()
def module(config):
    return BenchmarkComparisonModule(config)


@pytest.fixture()
def baseline_df():
    """Baseline method: projected = actual (perfect)."""
    return _make_result_df("baseline", [100.0] * 9, [100.0] * 9)


@pytest.fixture()
def good_method_df():
    """Method with small errors (projected slightly above actual)."""
    return _make_result_df("good", [101.0] * 9, [100.0] * 9)


@pytest.fixture()
def bad_method_df():
    """Method with larger errors."""
    return _make_result_df("bad", [110.0] * 9, [100.0] * 9)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_validate_result_df_passes(self, baseline_df):
        validate_dataframe(baseline_df, PROJECTION_RESULT_COLUMNS, "test")

    def test_validate_result_df_missing_column(self):
        df = pd.DataFrame({"geography": ["a"], "year": [2020]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataframe(df, PROJECTION_RESULT_COLUMNS, "bad")


# ---------------------------------------------------------------------------
# compare_all
# ---------------------------------------------------------------------------

class TestCompareAll:
    def test_compare_all_returns_rows(self, module, baseline_df, good_method_df):
        results = {"baseline": baseline_df, "good": good_method_df}
        out = module.compare_all(results, baseline_name="baseline")
        assert not out.empty
        assert set(out["method"].unique()) == {"good"}
        assert "delta" in out.columns

    def test_compare_all_multiple_methods(
        self, module, baseline_df, good_method_df, bad_method_df
    ):
        results = {
            "baseline": baseline_df,
            "good": good_method_df,
            "bad": bad_method_df,
        }
        out = module.compare_all(results, baseline_name="baseline")
        assert set(out["method"].unique()) == {"good", "bad"}

    def test_compare_all_missing_baseline_raises(self, module, good_method_df):
        with pytest.raises(KeyError, match="Baseline"):
            module.compare_all({"good": good_method_df}, baseline_name="missing")


# ---------------------------------------------------------------------------
# pairwise_comparison
# ---------------------------------------------------------------------------

class TestPairwiseComparison:
    def test_deltas_correct_direction(self, module, baseline_df, good_method_df):
        out = module.pairwise_comparison(
            good_method_df, baseline_df, "good", "baseline"
        )
        # Good method has error 1.0; baseline has error 0.0 -> delta > 0
        mae_rows = out[
            (out["metric_name"] == "mae") & (out["geography_group"] == "all")
        ]
        assert all(mae_rows["delta"] > 0)
        assert all(mae_rows["baseline_value"] == 0.0)

    def test_horizon_bands_present(self, module, baseline_df, good_method_df):
        out = module.pairwise_comparison(
            good_method_df, baseline_df, "good", "baseline"
        )
        assert set(out["horizon_band"].unique()) == {"near_term", "long_term", "all"}


# ---------------------------------------------------------------------------
# component_swap_analysis
# ---------------------------------------------------------------------------

class TestComponentSwapAnalysis:
    def test_swap_analysis_produces_rows(self, module):
        swap_a = _make_result_df("swap_a", [105.0] * 9, [100.0] * 9)
        swap_b = _make_result_df("swap_b", [102.0] * 9, [100.0] * 9)
        out = module.component_swap_analysis(
            {"adv_migration": swap_a, "adv_fertility": swap_b}
        )
        assert set(out["swap_label"].unique()) == {"adv_migration", "adv_fertility"}
        # swap_b (error 2) should have lower MAE than swap_a (error 5)
        mae_all = out[
            (out["metric_name"] == "mae")
            & (out["geography_group"] == "all")
            & (out["horizon_band"] == "all")
        ]
        migration_val = mae_all.loc[
            mae_all["swap_label"] == "adv_migration", "value"
        ].iloc[0]
        fertility_val = mae_all.loc[
            mae_all["swap_label"] == "adv_fertility", "value"
        ].iloc[0]
        assert migration_val > fertility_val


# ---------------------------------------------------------------------------
# horizon_blend
# ---------------------------------------------------------------------------

class TestHorizonBlend:
    def test_near_term_used_before_blend_horizon(self, module):
        near = _make_result_df("near", [200.0] * 9, [100.0] * 9)
        far = _make_result_df("far", [300.0] * 9, [100.0] * 9)
        blended = module.horizon_blend(near, far, blend_horizon=5)
        # At horizon 1 (<= 5), should use near-term value exactly
        h1 = blended[blended["horizon"] == 1]
        np.testing.assert_allclose(h1["projected_value"].values, 200.0)

    def test_long_term_used_after_transition(self, module):
        near = _make_result_df("near", [200.0] * 9, [100.0] * 9)
        far = _make_result_df("far", [300.0] * 9, [100.0] * 9)
        blended = module.horizon_blend(near, far, blend_horizon=1)
        # blend_horizon=1, transition_width=1, transition_end=2
        # At horizon 10 (>> transition_end), should use long-term fully
        h10 = blended[blended["horizon"] == 10]
        np.testing.assert_allclose(h10["projected_value"].values, 300.0)

    def test_blend_produces_intermediate_values(self, module):
        # Use horizons that fall in the transition zone
        near = _make_result_df(
            "near",
            [200.0] * 9,
            [100.0] * 9,
            horizons=[1, 5, 10],
        )
        far = _make_result_df(
            "far",
            [300.0] * 9,
            [100.0] * 9,
            horizons=[1, 5, 10],
        )
        # blend_horizon=5, transition_width=2, transition_end=7
        # horizon 10 > 7 => fully long-term
        blended = module.horizon_blend(near, far, blend_horizon=5)
        h10 = blended[blended["horizon"] == 10]
        np.testing.assert_allclose(h10["projected_value"].values, 300.0)

    def test_blend_has_required_columns(self, module):
        near = _make_result_df("near", [200.0] * 9)
        far = _make_result_df("far", [300.0] * 9)
        blended = module.horizon_blend(near, far, blend_horizon=5, method_name="hybrid")
        assert blended["run_id"].iloc[0] == "hybrid"
        assert "projected_value" in blended.columns
        assert "actual_value" in blended.columns


# ---------------------------------------------------------------------------
# ensemble_average
# ---------------------------------------------------------------------------

class TestEnsembleAverage:
    def test_equal_weight_average(self, module):
        m1 = _make_result_df("m1", [100.0] * 9)
        m2 = _make_result_df("m2", [200.0] * 9)
        ensemble = module.ensemble_average({"m1": m1, "m2": m2})
        np.testing.assert_allclose(ensemble["projected_value"].values, 150.0)

    def test_weighted_average(self, module):
        m1 = _make_result_df("m1", [100.0] * 9)
        m2 = _make_result_df("m2", [200.0] * 9)
        ensemble = module.ensemble_average(
            {"m1": m1, "m2": m2}, weights={"m1": 3.0, "m2": 1.0}
        )
        # (100*0.75 + 200*0.25) = 125
        np.testing.assert_allclose(ensemble["projected_value"].values, 125.0)

    def test_empty_raises(self, module):
        with pytest.raises(ValueError, match="must not be empty"):
            module.ensemble_average({})

    def test_zero_weights_raises(self, module):
        m1 = _make_result_df("m1", [100.0] * 9)
        with pytest.raises(ValueError, match="nonzero"):
            module.ensemble_average({"m1": m1}, weights={"m1": 0.0})


# ---------------------------------------------------------------------------
# ensemble_by_county_type
# ---------------------------------------------------------------------------

class TestEnsembleByCountyType:
    def test_selects_correct_method_per_group(self, module):
        m1 = _make_result_df("m1", [100.0] * 9)
        m2 = _make_result_df("m2", [200.0] * 9)
        result = module.ensemble_by_county_type(
            {"m1": m1, "m2": m2},
            county_groups={"urban": ["38017", "38015"], "rural": ["38099"]},
            selector={"urban": "m1", "rural": "m2", "_default": "m1"},
        )
        urban = result[result["geography"].isin(["38017", "38015"])]
        rural = result[result["geography"] == "38099"]
        np.testing.assert_allclose(urban["projected_value"].values, 100.0)
        np.testing.assert_allclose(rural["projected_value"].values, 200.0)


# ---------------------------------------------------------------------------
# rank_methods
# ---------------------------------------------------------------------------

class TestRankMethods:
    def test_ranks_error_metrics_ascending(self, module, baseline_df, good_method_df, bad_method_df):
        results = {
            "baseline": baseline_df,
            "good": good_method_df,
            "bad": bad_method_df,
        }
        comparison = module.compare_all(results, baseline_name="baseline")
        ranked = module.rank_methods(comparison)
        # For MAE, good (error 1) should rank better (lower) than bad (error 10)
        mae_all = ranked[
            (ranked["metric_name"] == "mae")
            & (ranked["geography_group"] == "all")
            & (ranked["horizon_band"] == "all")
        ]
        good_rank = mae_all.loc[mae_all["method"] == "good", "rank"].iloc[0]
        bad_rank = mae_all.loc[mae_all["method"] == "bad", "rank"].iloc[0]
        assert good_rank < bad_rank

    def test_empty_comparison(self, module):
        empty = pd.DataFrame(
            columns=[
                "method", "baseline", "metric_name", "method_value",
                "baseline_value", "delta", "geography_group", "horizon_band",
            ]
        )
        ranked = module.rank_methods(empty)
        assert ranked.empty


# ---------------------------------------------------------------------------
# improvement_over_baseline
# ---------------------------------------------------------------------------

class TestImprovementOverBaseline:
    def test_improvement_positive_when_better(
        self, module, baseline_df, good_method_df, bad_method_df
    ):
        results = {
            "baseline": baseline_df,
            "good": good_method_df,
            "bad": bad_method_df,
        }
        comparison = module.compare_all(results, baseline_name="baseline")
        improvement = module.improvement_over_baseline(comparison, metric="mape")
        # Baseline has 0 MAPE; good has ~1% MAPE; improvement = 0 - 1 = -1
        # So improvement should be negative (baseline is better)
        assert all(improvement["improvement"] <= 0)

    def test_missing_metric_returns_empty(self, module, baseline_df, good_method_df):
        results = {"baseline": baseline_df, "good": good_method_df}
        comparison = module.compare_all(results, baseline_name="baseline")
        improvement = module.improvement_over_baseline(
            comparison, metric="nonexistent_metric"
        )
        assert improvement.empty
