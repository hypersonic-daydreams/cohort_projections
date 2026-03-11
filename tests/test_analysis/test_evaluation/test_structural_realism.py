"""Tests for Module 2: Structural Realism checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cohort_projections.analysis.evaluation.structural_realism import (
    _STANDARD_AGE_GROUPS,
    StructuralRealismModule,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic data builders
# ---------------------------------------------------------------------------

def _make_components_df(
    *,
    n_geos: int = 3,
    horizons: tuple[int, ...] = (5,),
    components: tuple[str, ...] = ("births", "deaths", "net_migration"),
    error_scale: float = 0.0,
) -> pd.DataFrame:
    """Build a synthetic ComponentRecord-compatible DataFrame.

    When ``error_scale == 0``, projected == actual (perfect forecast).
    """
    rng = np.random.default_rng(42)
    rows = []
    for g in range(n_geos):
        geo = f"3800{g}"
        for h in horizons:
            year = 2020 + h
            for comp in components:
                actual = rng.uniform(100, 500)
                projected = actual * (1 + rng.normal(0, error_scale))
                rows.append({
                    "run_id": "test-run",
                    "geography": geo,
                    "year": year,
                    "horizon": h,
                    "component": comp,
                    "projected_component_value": projected,
                    "actual_component_value": actual,
                })
    return pd.DataFrame(rows)


def _make_results_df(
    *,
    geos: list[str] | None = None,
    years: list[int] | None = None,
    include_totals: bool = True,
    include_state: bool = False,
    age_error_scale: float = 0.0,
    pop_base: float = 10000.0,
) -> pd.DataFrame:
    """Build a synthetic ProjectionResultRecord-compatible DataFrame.

    Produces rows for each geography/year/sex/age_group combination.
    """
    rng = np.random.default_rng(99)
    if geos is None:
        geos = ["38001", "38002", "38003"]
    if years is None:
        years = [2025, 2030]
    origin = min(years)

    rows = []

    all_geos = list(geos)
    geo_types = dict.fromkeys(geos, "county")
    if include_state:
        all_geos.append("state")
        geo_types["state"] = "state"

    for geo in all_geos:
        for year in years:
            for sex in ("male", "female"):
                age_values_actual: list[float] = []
                age_values_proj: list[float] = []
                base_val = pop_base / len(_STANDARD_AGE_GROUPS)
                for ag in _STANDARD_AGE_GROUPS:
                    actual = base_val + rng.normal(0, base_val * 0.1)
                    actual = max(actual, 10.0)
                    projected = actual * (1 + rng.normal(0, age_error_scale))
                    projected = max(projected, 0.0)
                    age_values_actual.append(actual)
                    age_values_proj.append(projected)
                    rows.append({
                        "run_id": "test-run",
                        "geography": geo,
                        "geography_type": geo_types[geo],
                        "year": year,
                        "horizon": year - origin,
                        "sex": sex,
                        "age_group": ag,
                        "target": "population",
                        "projected_value": projected,
                        "actual_value": actual,
                        "base_value": actual * 0.95,
                    })

                if include_totals:
                    rows.append({
                        "run_id": "test-run",
                        "geography": geo,
                        "geography_type": geo_types[geo],
                        "year": year,
                        "horizon": year - origin,
                        "sex": sex,
                        "age_group": "total",
                        "target": "population",
                        "projected_value": sum(age_values_proj),
                        "actual_value": sum(age_values_actual),
                        "base_value": sum(age_values_actual) * 0.95,
                    })

            # Add sex=total rows
            if include_totals:
                sex_total_proj = 0.0
                sex_total_act = 0.0
                for sex_val in ("male", "female"):
                    sex_rows = [
                        r for r in rows
                        if r["geography"] == geo
                        and r["year"] == year
                        and r["sex"] == sex_val
                        and r["age_group"] == "total"
                    ]
                    if sex_rows:
                        sex_total_proj += sex_rows[0]["projected_value"]
                        sex_total_act += sex_rows[0]["actual_value"]

                rows.append({
                    "run_id": "test-run",
                    "geography": geo,
                    "geography_type": geo_types[geo],
                    "year": year,
                    "horizon": year - origin,
                    "sex": "total",
                    "age_group": "total",
                    "target": "population",
                    "projected_value": sex_total_proj,
                    "actual_value": sex_total_act,
                    "base_value": sex_total_act * 0.95,
                })

    # If state included, set state total = sum of county totals
    if include_state:
        df_tmp = pd.DataFrame(rows)
        for year in years:
            county_total = df_tmp[
                (df_tmp["geography_type"] == "county")
                & (df_tmp["year"] == year)
                & (df_tmp["sex"] == "total")
                & (df_tmp["age_group"] == "total")
            ]
            state_proj = county_total["projected_value"].sum()
            state_act = county_total["actual_value"].sum()
            # Update the state row
            for r in rows:
                if (
                    r["geography"] == "state"
                    and r["year"] == year
                    and r["sex"] == "total"
                    and r["age_group"] == "total"
                ):
                    r["projected_value"] = state_proj
                    r["actual_value"] = state_act
                    r["base_value"] = state_act * 0.95

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def module() -> StructuralRealismModule:
    return StructuralRealismModule()


class TestComponentRealism:
    """Tests for component_realism()."""

    def test_perfect_components(self, module: StructuralRealismModule) -> None:
        """Perfect forecast should yield zero MAPE."""
        df = _make_components_df(error_scale=0.0)
        result = module.component_realism(df)
        assert not result.empty
        mape_rows = result[result["metric_name"].str.endswith("_mape")]
        assert (mape_rows["value"] == 0.0).all()

    def test_noisy_components(self, module: StructuralRealismModule) -> None:
        """Noisy forecast should yield positive MAPE."""
        df = _make_components_df(error_scale=0.2)
        result = module.component_realism(df)
        mape_rows = result[result["metric_name"].str.endswith("_mape")]
        assert (mape_rows["value"] > 0).all()

    def test_empty_components(self, module: StructuralRealismModule) -> None:
        """Empty input returns empty output."""
        result = module.component_realism(pd.DataFrame())
        assert result.empty

    def test_multiple_horizons(self, module: StructuralRealismModule) -> None:
        """Each horizon produces separate diagnostic rows."""
        df = _make_components_df(horizons=(5, 10, 15))
        result = module.component_realism(df)
        horizons = result["horizon"].dropna().unique()
        assert set(horizons) == {5, 10, 15}


class TestAgeStructureRealism:
    """Tests for age_structure_realism()."""

    def test_perfect_age_structure(self, module: StructuralRealismModule) -> None:
        """Identical projected/actual age distributions have JSD ~ 0."""
        df = _make_results_df(age_error_scale=0.0)
        result = module.age_structure_realism(df)
        jsd_rows = result[result["metric_name"] == "age_jsd"]
        assert not jsd_rows.empty
        assert (jsd_rows["value"] < 1e-10).all()

    def test_divergent_age_structure(self, module: StructuralRealismModule) -> None:
        """Large age errors produce positive JSD, possibly with WARN flags."""
        df = _make_results_df(age_error_scale=0.5)
        result = module.age_structure_realism(df)
        jsd_rows = result[result["metric_name"] == "age_jsd"]
        assert (jsd_rows["value"] > 0).all()

    def test_roughness_computed(self, module: StructuralRealismModule) -> None:
        """Age-schedule roughness metric is produced."""
        df = _make_results_df()
        result = module.age_structure_realism(df)
        roughness = result[result["metric_name"] == "age_roughness"]
        assert not roughness.empty


class TestCohortContinuity:
    """Tests for cohort_continuity()."""

    def test_stable_cohorts(self, module: StructuralRealismModule) -> None:
        """Cohorts with stable populations produce small residuals."""
        df = _make_results_df(years=[2020, 2025, 2030], age_error_scale=0.0)
        result = module.cohort_continuity(df)
        assert not result.empty
        # With stable synthetic data, residuals should exist
        assert "cohort_survival_residual" in result["metric_name"].values

    def test_single_year_no_transitions(self, module: StructuralRealismModule) -> None:
        """A single year cannot produce cohort transitions."""
        df = _make_results_df(years=[2025])
        result = module.cohort_continuity(df)
        assert result.empty


class TestAccountingChecks:
    """Tests for accounting_checks()."""

    def test_age_sex_totals_consistent(self, module: StructuralRealismModule) -> None:
        """When detail sums match totals, mismatch should be ~0."""
        df = _make_results_df(include_totals=True)
        result = module.accounting_checks(df, pd.DataFrame())
        mismatch = result[result["metric_name"] == "age_sex_total_mismatch_pct"]
        assert not mismatch.empty
        # Our fixture constructs totals as exact sums, so mismatch ~ 0
        assert (mismatch["value"] < 0.01).all()

    def test_county_state_consistency(self, module: StructuralRealismModule) -> None:
        """County totals should match state total when constructed correctly."""
        df = _make_results_df(include_totals=True, include_state=True)
        result = module.accounting_checks(df, pd.DataFrame())
        state_check = result[
            result["metric_name"] == "county_state_total_mismatch_pct"
        ]
        assert not state_check.empty
        assert (state_check["value"] < 0.01).all()

    def test_broken_county_state(self, module: StructuralRealismModule) -> None:
        """Deliberately wrong state total should produce large mismatch."""
        df = _make_results_df(include_totals=True, include_state=True)
        # Corrupt state total
        mask = (
            (df["geography"] == "state")
            & (df["sex"] == "total")
            & (df["age_group"] == "total")
        )
        df.loc[mask, "projected_value"] = 999999.0
        result = module.accounting_checks(df, pd.DataFrame())
        state_check = result[
            result["metric_name"] == "county_state_total_mismatch_pct"
        ]
        assert not state_check.empty
        # At least one FAIL
        assert any("FAIL" in str(n) for n in state_check["notes"])

    def test_empty_inputs(self, module: StructuralRealismModule) -> None:
        """Empty DataFrames return empty diagnostics."""
        result = module.accounting_checks(pd.DataFrame(), pd.DataFrame())
        assert result.empty


class TestDistributionalRealism:
    """Tests for distributional_realism()."""

    def test_produces_diagnostics(self, module: StructuralRealismModule) -> None:
        """Non-trivial data should produce variance ratio and skewness checks."""
        df = _make_results_df(
            geos=[f"380{i:02d}" for i in range(10)],
            include_totals=True,
        )
        result = module.distributional_realism(df)
        assert not result.empty
        metric_names = set(result["metric_name"])
        assert "county_size_dist_jsd" in metric_names
        assert "county_size_variance_ratio" in metric_names

    def test_perfect_distribution(self, module: StructuralRealismModule) -> None:
        """When projected == actual, JSD should be ~0 and variance ratio ~1."""
        df = _make_results_df(
            geos=[f"380{i:02d}" for i in range(10)],
            include_totals=True,
            age_error_scale=0.0,
        )
        result = module.distributional_realism(df)
        jsd = result[result["metric_name"] == "county_size_dist_jsd"]
        assert (jsd["value"] < 1e-6).all()

        var_ratio = result[result["metric_name"] == "county_size_variance_ratio"]
        assert all(abs(v - 1.0) < 0.01 for v in var_ratio["value"])


class TestComputeAllChecks:
    """Tests for the top-level compute_all_checks()."""

    def test_combines_all_modules(self, module: StructuralRealismModule) -> None:
        """compute_all_checks produces diagnostics from multiple sub-modules."""
        results_df = _make_results_df(
            geos=[f"380{i:02d}" for i in range(5)],
            years=[2020, 2025, 2030],
            include_totals=True,
            include_state=True,
            age_error_scale=0.05,
        )
        components_df = _make_components_df(
            n_geos=5, horizons=(5, 10), error_scale=0.1,
        )
        result = module.compute_all_checks(results_df, components_df)
        assert not result.empty
        # Should have diagnostics from multiple metric groups
        metric_prefixes = {m.split("_")[0] for m in result["metric_name"]}
        # At minimum we expect age, cohort, and component metrics
        assert len(metric_prefixes) >= 3

    def test_diagnostic_record_columns(self, module: StructuralRealismModule) -> None:
        """Output DataFrame has columns compatible with DiagnosticRecord."""
        results_df = _make_results_df(include_totals=True)
        components_df = _make_components_df()
        result = module.compute_all_checks(results_df, components_df)
        expected_cols = {
            "run_id", "metric_name", "metric_group", "geography",
            "geography_group", "target", "horizon", "value",
            "comparison_run_id", "notes",
        }
        assert expected_cols.issubset(set(result.columns))
