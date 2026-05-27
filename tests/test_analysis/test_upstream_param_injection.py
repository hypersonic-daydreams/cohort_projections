"""Tests for upstream parameter injection (GQ fraction, rate cap) in walk-forward validation.

Verifies that:
1. Default upstream parameter values produce rates identical to pre-computed rates.
2. Overridden values produce different rates.
3. Config injection handles the new float keys correctly.
4. Rate cap is applied only when config differs from the default.
5. GQ correction recomputation is triggered only when fraction differs from 1.0.
"""

# ruff: noqa: I001

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))

import walk_forward_validation as wfv  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_rate_series() -> tuple[pd.Series, pd.Series]:
    """Return a rate series and corresponding age_group series for testing."""
    age_groups = pd.Series(
        [
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
    )
    rates = pd.Series(
        [
            0.02,
            0.01,
            -0.01,
            0.12,
            -0.10,
            0.05,
            0.03,
            0.02,
            0.01,
            -0.005,
            -0.01,
            -0.02,
            -0.03,
            -0.04,
            -0.05,
            -0.06,
            -0.07,
            -0.09,
        ]
    )
    return rates, age_groups


@pytest.fixture()
def convergence_windows() -> dict[str, pd.DataFrame]:
    """Return minimal convergence windows for 2 counties."""
    counties = ["38001", "38017"]
    age_groups = ["0-4", "15-19", "20-24", "30-34"]
    sexes = ["Male", "Female"]
    rows = []
    for fips in counties:
        for ag in age_groups:
            for sex in sexes:
                rows.append(
                    {
                        "county_fips": fips,
                        "age_group": ag,
                        "sex": sex,
                        "migration_rate_annual": np.random.default_rng(42).normal(0.02, 0.05),
                    }
                )

    recent = pd.DataFrame(rows)
    medium = recent.copy()
    medium["migration_rate_annual"] *= 0.8
    longterm = recent.copy()
    longterm["migration_rate_annual"] *= 0.6
    return {"recent": recent, "medium": medium, "longterm": longterm}


# ---------------------------------------------------------------------------
# Test _apply_annual_rate_cap
# ---------------------------------------------------------------------------


class TestApplyAnnualRateCap:
    """Tests for the _apply_annual_rate_cap function."""

    def test_default_cap_clips_general_ages(self, sample_rate_series: tuple) -> None:
        rates, age_groups = sample_rate_series
        capped = wfv._apply_annual_rate_cap(rates, age_groups, general_cap=0.08)
        # 85+ has rate -0.09 which should be clipped to -0.08
        idx_85plus = age_groups[age_groups == "85+"].index[0]
        assert abs(capped.iloc[idx_85plus] - (-0.08)) < 1e-9

    def test_college_ages_use_wider_cap(self, sample_rate_series: tuple) -> None:
        rates, age_groups = sample_rate_series
        capped = wfv._apply_annual_rate_cap(rates, age_groups, general_cap=0.08)
        # 15-19 has rate 0.12 which exceeds general_cap but is within college_cap (0.15)
        idx_15_19 = age_groups[age_groups == "15-19"].index[0]
        assert abs(capped.iloc[idx_15_19] - 0.12) < 1e-9

    def test_tighter_cap_clips_more(self, sample_rate_series: tuple) -> None:
        rates, age_groups = sample_rate_series
        capped_08 = wfv._apply_annual_rate_cap(rates, age_groups, general_cap=0.08)
        capped_06 = wfv._apply_annual_rate_cap(rates, age_groups, general_cap=0.06)
        # 0.06 cap should clip more cells than 0.08
        n_clipped_08 = (capped_08 != rates).sum()
        n_clipped_06 = (capped_06 != rates).sum()
        assert n_clipped_06 >= n_clipped_08

    def test_no_cap_at_zero_clips_everything(self, sample_rate_series: tuple) -> None:
        rates, age_groups = sample_rate_series
        capped = wfv._apply_annual_rate_cap(rates, age_groups, general_cap=0.0, college_cap=0.0)
        # Everything should be clipped to 0
        assert (capped == 0.0).all()


# ---------------------------------------------------------------------------
# Test get_convergence_rate_for_year with rate cap
# ---------------------------------------------------------------------------


class TestConvergenceRateCapIntegration:
    """Tests for rate cap application within get_convergence_rate_for_year."""

    def test_default_cap_no_change(self, convergence_windows: dict) -> None:
        """Default rate_cap_general (0.08) should NOT trigger capping."""
        cfg: wfv.MethodConfig = {
            "convergence_recent_hold": 1,
            "convergence_medium_hold": 2,
            "convergence_transition_hold": 1,
            "rate_cap_general": wfv._DEFAULT_RATE_CAP_GENERAL,
        }
        result_with = wfv.get_convergence_rate_for_year(3, convergence_windows, cfg)
        result_without = wfv.get_convergence_rate_for_year(
            3,
            convergence_windows,
            {
                "convergence_recent_hold": 1,
                "convergence_medium_hold": 2,
                "convergence_transition_hold": 1,
            },
        )
        pd.testing.assert_frame_equal(result_with, result_without)

    def test_non_default_cap_triggers_capping(self, convergence_windows: dict) -> None:
        """A non-default rate_cap_general should actually clip rates."""
        cfg: wfv.MethodConfig = {
            "convergence_recent_hold": 1,
            "convergence_medium_hold": 2,
            "convergence_transition_hold": 1,
            "rate_cap_general": 0.01,  # very tight cap
        }
        result = wfv.get_convergence_rate_for_year(3, convergence_windows, cfg)
        # All non-college rates should be within [-0.01, 0.01]
        non_college = result[~result["age_group"].isin(["15-19", "20-24"])]
        assert non_college["migration_rate_annual"].abs().max() <= 0.01 + 1e-9


# ---------------------------------------------------------------------------
# Test maybe_recompute_mig_raw
# ---------------------------------------------------------------------------


class TestMaybeRecomputeMigRaw:
    """Tests for the maybe_recompute_mig_raw function."""

    def test_default_fraction_returns_original(self) -> None:
        """Default gq_correction_fraction (1.0) should return original mig_raw."""
        mig_raw = pd.DataFrame(
            {
                "county_fips": ["38001"],
                "age_group": ["0-4"],
                "sex": ["Male"],
                "period_start": [2000],
                "period_end": [2005],
                "migration_rate": [0.01],
            }
        )
        cfg: wfv.MethodConfig = {
            "gq_correction_fraction": wfv._DEFAULT_GQ_CORRECTION_FRACTION,
        }
        result = wfv.maybe_recompute_mig_raw(mig_raw, {}, cfg)
        assert result is mig_raw  # exact same object

    def test_missing_key_returns_original(self) -> None:
        """Missing gq_correction_fraction key should return original mig_raw."""
        mig_raw = pd.DataFrame(
            {
                "county_fips": ["38001"],
                "age_group": ["0-4"],
                "sex": ["Male"],
                "period_start": [2000],
                "period_end": [2005],
                "migration_rate": [0.01],
            }
        )
        cfg: wfv.MethodConfig = {}
        result = wfv.maybe_recompute_mig_raw(mig_raw, {}, cfg)
        assert result is mig_raw

    @patch("walk_forward_validation.recompute_migration_with_gq_override")
    def test_non_default_fraction_triggers_recompute(self, mock_recompute: MagicMock) -> None:
        """Non-default gq_correction_fraction should trigger recomputation."""
        mig_raw = pd.DataFrame(
            {
                "county_fips": ["38001"],
                "age_group": ["0-4"],
                "sex": ["Male"],
                "period_start": [2000],
                "period_end": [2005],
                "migration_rate": [0.01],
            }
        )
        mock_recompute.return_value = mig_raw.copy()
        cfg: wfv.MethodConfig = {"gq_correction_fraction": 0.75}
        snapshots = {2000: pd.DataFrame()}

        wfv.maybe_recompute_mig_raw(mig_raw, snapshots, cfg)

        mock_recompute.assert_called_once_with(snapshots, 0.75)

    def test_gq_recompute_collapses_full_survival_csv_to_age_groups(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GQ override recomputation should pass age-group survival rates downstream."""
        survival_full = pd.DataFrame(
            {
                "age": [0, 5, 0, 5],
                "sex": ["Male", "Male", "Female", "Female"],
                "survival_rate_5yr": [0.99, 0.98, 0.995, 0.985],
            }
        )
        snapshots = {
            2000: pd.DataFrame({"county_fips": ["38001"], "age_group": ["0-4"], "sex": ["Male"], "population": [10]}),
            2005: pd.DataFrame({"county_fips": ["38001"], "age_group": ["5-9"], "sex": ["Male"], "population": [9]}),
        }
        captured_survival: list[pd.DataFrame] = []

        monkeypatch.setattr(wfv, "ALL_PERIODS", [(2000, 2005)])
        monkeypatch.setattr(wfv.pd, "read_csv", lambda path: survival_full)
        monkeypatch.setattr(wfv.pd, "read_parquet", lambda path: pd.DataFrame())

        import cohort_projections.data.process.residual_migration as rm

        monkeypatch.setattr(
            rm,
            "subtract_gq_from_populations",
            lambda snapshots, gq_historical, fraction: snapshots,
        )

        def fake_compute_residual_migration_rates(
            *,
            pop_start: pd.DataFrame,
            pop_end: pd.DataFrame,
            survival_rates: pd.DataFrame,
            period: tuple[int, int],
        ) -> pd.DataFrame:
            del pop_start, pop_end, period
            captured_survival.append(survival_rates.copy())
            return pd.DataFrame(
                {
                    "county_fips": ["38001"],
                    "age_group": ["5-9"],
                    "sex": ["Male"],
                    "period_start": [2000],
                    "period_end": [2005],
                    "migration_rate": [0.01],
                }
            )

        monkeypatch.setattr(rm, "compute_residual_migration_rates", fake_compute_residual_migration_rates)

        result = wfv.recompute_migration_with_gq_override(snapshots, 0.75)

        assert list(result.columns) == [
            "county_fips",
            "age_group",
            "sex",
            "period_start",
            "period_end",
            "migration_rate",
        ]
        assert captured_survival
        assert set(captured_survival[0].columns) == {"age_group", "sex", "survival_rate_5yr"}
        assert set(captured_survival[0]["age_group"]) == {"0-4", "5-9"}


# ---------------------------------------------------------------------------
# Test MethodConfig defaults
# ---------------------------------------------------------------------------


class TestMethodConfigDefaults:
    """Tests that upstream parameters are present in method config dicts."""

    def test_m2026_config_has_upstream_params(self) -> None:
        assert "gq_correction_fraction" in wfv.M2026_CONFIG
        assert "rate_cap_general" in wfv.M2026_CONFIG
        assert wfv.M2026_CONFIG["gq_correction_fraction"] == 1.0
        assert wfv.M2026_CONFIG["rate_cap_general"] == 0.08

    def test_m2026r1_inherits_upstream_params(self) -> None:
        assert "gq_correction_fraction" in wfv.M2026R1_CONFIG
        assert "rate_cap_general" in wfv.M2026R1_CONFIG
        assert wfv.M2026R1_CONFIG["gq_correction_fraction"] == 1.0
        assert wfv.M2026R1_CONFIG["rate_cap_general"] == 0.08

    def test_method_dispatch_configs_have_upstream_params(self) -> None:
        for method_id in ["m2026", "m2026r1"]:
            cfg = wfv.METHOD_DISPATCH[method_id]["config"]
            assert "gq_correction_fraction" in cfg  # type: ignore[operator]
            assert "rate_cap_general" in cfg  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Test config injection round-trip (YAML → METHOD_DISPATCH → rate computation)
# ---------------------------------------------------------------------------


class TestConfigInjectionRoundTrip:
    """Test that YAML config_delta flows through to rate computation."""

    def test_yaml_float_keys_pass_through(self) -> None:
        """Float keys in resolved_config should pass through _yaml_config_to_method_config."""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
        import run_benchmark_suite as rbs

        resolved_config = {
            "gq_correction_fraction": 0.75,
            "rate_cap_general": 0.06,
            "college_blend_factor": 0.7,
        }
        result = rbs._yaml_config_to_method_config(resolved_config)
        assert result["gq_correction_fraction"] == 0.75
        assert result["rate_cap_general"] == 0.06
        assert result["college_blend_factor"] == 0.7


class TestSearchOnlyBenchmarkKnobs:
    """Tests for sandbox-only benchmark knobs exposed through MethodConfig."""

    def test_default_period_sets_match_current_behavior(self) -> None:
        assert wfv._resolve_convergence_period_sets(2005) == {
            "recent": [(2000, 2005)],
            "medium": [(2000, 2005)],
            "longterm": [(2000, 2005)],
        }
        assert wfv._resolve_convergence_period_sets(2015) == {
            "recent": [(2010, 2015)],
            "medium": [(2005, 2010), (2010, 2015)],
            "longterm": [(2000, 2005), (2005, 2010), (2010, 2015)],
        }
        assert wfv._resolve_convergence_period_sets(2020) == {
            "recent": [(2015, 2020)],
            "medium": [(2010, 2015), (2015, 2020)],
            "longterm": [
                (2000, 2005),
                (2005, 2010),
                (2010, 2015),
                (2015, 2020),
            ],
        }

    def test_recent_window_override_expands_period_sets(self) -> None:
        cfg: wfv.MethodConfig = {
            "convergence_recent_period_count": 2,
            "convergence_medium_period_count": 3,
        }
        assert wfv._resolve_convergence_period_sets(2020, cfg) == {
            "recent": [(2010, 2015), (2015, 2020)],
            "medium": [(2005, 2010), (2010, 2015), (2015, 2020)],
            "longterm": [
                (2000, 2005),
                (2005, 2010),
                (2010, 2015),
                (2015, 2020),
            ],
        }

    def test_custom_mortality_improvement_rate_changes_survival_schedule(self) -> None:
        base_survival = {("0-4", "Male"): 0.99}
        default_schedule = wfv._get_improved_survival_annual(base_survival, 5)
        conservative_cfg: wfv.MethodConfig = {"mortality_improvement_rate": 0.003}
        conservative_schedule = wfv._get_improved_survival_annual(
            base_survival,
            5,
            conservative_cfg,
        )
        assert conservative_schedule[("0-4", "Male")] < default_schedule[("0-4", "Male")]

    def test_set_keys_still_converted(self) -> None:
        """Set keys should still be converted from lists to sets."""
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
        import run_benchmark_suite as rbs

        resolved_config = {
            "bakken_fips": ["38105", "38053"],
            "gq_correction_fraction": 0.75,
        }
        result = rbs._yaml_config_to_method_config(resolved_config)
        assert isinstance(result["bakken_fips"], set)
        assert isinstance(result["gq_correction_fraction"], float)
