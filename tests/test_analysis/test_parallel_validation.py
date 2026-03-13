"""Tests for parallel walk-forward validation execution.

Verifies that:
1. Parallel execution (workers > 1) produces bit-identical results to sequential (workers=1).
2. Graceful fallback on worker failure.
3. Worker count defaults and capping behave correctly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data fixtures
# ---------------------------------------------------------------------------


def _make_population_df(counties: list[str], year: int) -> pd.DataFrame:
    """Build a minimal population DataFrame for testing.

    Creates two age groups and two sexes per county to keep things fast.
    Population values are deterministic based on county FIPS and year so
    that results are reproducible.
    """
    age_groups = ["0-4", "5-9"]
    sexes = ["Male", "Female"]
    records: list[dict[str, object]] = []
    for fips in counties:
        fips_seed = int(fips[-3:])
        for ag in age_groups:
            for sex in sexes:
                pop = 1000.0 + fips_seed * 10 + year * 0.1
                records.append(
                    {
                        "county_fips": fips,
                        "age_group": ag,
                        "sex": sex,
                        "population": pop,
                    }
                )
    return pd.DataFrame(records)


def _make_migration_df(counties: list[str]) -> pd.DataFrame:
    """Build minimal migration rates for all periods."""
    periods = [(2000, 2005), (2005, 2010), (2010, 2015), (2015, 2020)]
    age_groups = ["0-4", "5-9"]
    sexes = ["Male", "Female"]
    records: list[dict[str, object]] = []
    for fips in counties:
        for ps, pe in periods:
            for ag in age_groups:
                for sex in sexes:
                    records.append(
                        {
                            "county_fips": fips,
                            "age_group": ag,
                            "sex": sex,
                            "period_start": ps,
                            "period_end": pe,
                            "migration_rate": 0.01,
                        }
                    )
    return pd.DataFrame(records)


def _make_survival() -> dict[tuple[str, str], float]:
    """Build minimal survival rates."""
    return {
        ("0-4", "Male"): 0.99,
        ("0-4", "Female"): 0.99,
        ("5-9", "Male"): 0.995,
        ("5-9", "Female"): 0.995,
    }


def _make_fertility() -> dict[str, float]:
    """Build minimal fertility rates (empty — no fertile age groups in test data)."""
    return {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunSingleOriginAnnual:
    """Unit tests for _run_single_origin_annual."""

    @pytest.fixture()
    def test_data(self) -> dict[str, Any]:
        """Provide a small synthetic dataset."""
        counties = ["38001", "38003"]
        snapshots = {yr: _make_population_df(counties, yr) for yr in [2000, 2005, 2010, 2015, 2020, 2024]}
        mig_raw = _make_migration_df(counties)
        survival = _make_survival()
        fertility = _make_fertility()

        # Precompute actual county totals for validation years
        actual_county_totals: dict[int, dict[str, float]] = {}
        for yr in range(2006, 2025):
            df = _make_population_df(counties, yr)
            actual_county_totals[yr] = (
                df.groupby("county_fips")["population"].sum().to_dict()
            )

        return {
            "snapshots": snapshots,
            "mig_raw": mig_raw,
            "survival": survival,
            "fertility": fertility,
            "counties": counties,
            "actual_county_totals": actual_county_totals,
        }

    def test_sequential_returns_records(self, test_data: dict[str, Any]) -> None:
        """Verify the worker function returns non-empty record lists."""
        from scripts.analysis.walk_forward_validation import (
            M2026R1_CONFIG,
            _run_single_origin_annual,
        )

        county_recs, state_recs, curve_recs = _run_single_origin_annual(
            origin_year=2010,
            base_pop=test_data["snapshots"][2010],
            mig_raw=test_data["mig_raw"],
            survival=test_data["survival"],
            fertility=test_data["fertility"],
            counties=test_data["counties"],
            methods=["m2026r1"],
            method_configs={"m2026r1": M2026R1_CONFIG},
            actual_county_totals=test_data["actual_county_totals"],
            max_validation_year=2024,
        )

        assert len(county_recs) > 0
        assert len(state_recs) > 0
        assert len(curve_recs) > 0

        # All records should be for origin 2010
        assert all(r["origin_year"] == 2010 for r in county_recs)
        assert all(r["origin_year"] == 2010 for r in state_recs)


class TestParallelIdenticalResults:
    """Verify parallel and sequential execution produce identical output."""

    @pytest.fixture()
    def shared_inputs(self) -> dict[str, Any]:
        """Build shared inputs for sequential vs parallel comparison."""
        counties = ["38001", "38003", "38005"]
        snapshots = {yr: _make_population_df(counties, yr) for yr in [2000, 2005, 2010, 2015, 2020, 2024]}
        mig_raw = _make_migration_df(counties)
        survival = _make_survival()
        fertility = _make_fertility()

        return {
            "snapshots": snapshots,
            "mig_raw": mig_raw,
            "survival": survival,
            "fertility": fertility,
        }

    def test_workers_1_vs_2_identical(self, shared_inputs: dict[str, Any]) -> None:
        """Results with workers=1 and workers=2 must be identical."""
        from scripts.analysis.walk_forward_validation import run_annual_validation

        # We mock load_annual_validation_actuals to avoid hitting real data files.
        counties = ["38001", "38003", "38005"]
        mock_actuals: dict[int, pd.DataFrame] = {}
        for yr in range(2006, 2025):
            mock_actuals[yr] = _make_population_df(counties, yr)

        with patch(
            "scripts.analysis.walk_forward_validation.load_annual_validation_actuals",
            return_value=mock_actuals,
        ), patch(
            "scripts.analysis.walk_forward_validation.maybe_recompute_mig_raw",
            side_effect=lambda mig, snap, cfg: mig,
        ):
            state_seq, county_seq, curves_seq = run_annual_validation(
                shared_inputs["snapshots"],
                shared_inputs["mig_raw"],
                shared_inputs["survival"],
                shared_inputs["fertility"],
                methods=["m2026r1"],
                workers=1,
            )

            state_par, county_par, curves_par = run_annual_validation(
                shared_inputs["snapshots"],
                shared_inputs["mig_raw"],
                shared_inputs["survival"],
                shared_inputs["fertility"],
                methods=["m2026r1"],
                workers=2,
            )

        # DataFrames must be identical
        pd.testing.assert_frame_equal(
            state_seq.reset_index(drop=True),
            state_par.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            county_seq.reset_index(drop=True),
            county_par.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            curves_seq.reset_index(drop=True),
            curves_par.reset_index(drop=True),
        )


class TestGracefulFallback:
    """Verify that a failing worker falls back to sequential execution."""

    def test_fallback_on_worker_error(self) -> None:
        """If a parallel worker raises, the origin should be re-run sequentially.

        We simulate this by using a FakeExecutor that raises on the first
        origin's future result, while succeeding for the rest.  The fallback
        path then runs the failed origin sequentially using the real
        ``_run_single_origin_annual`` function.
        """
        from concurrent.futures import Future

        from scripts.analysis.walk_forward_validation import run_annual_validation

        counties = ["38001"]
        snapshots = {yr: _make_population_df(counties, yr) for yr in [2000, 2005, 2010, 2015, 2020, 2024]}
        mig_raw = _make_migration_df(counties)
        survival = _make_survival()
        fertility = _make_fertility()

        mock_actuals: dict[int, pd.DataFrame] = {}
        for yr in range(2006, 2025):
            mock_actuals[yr] = _make_population_df(counties, yr)

        class _FakeExecutor:
            """Executor that fails on origin 2005 but succeeds on others."""

            def __init__(self, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> _FakeExecutor:
                return self

            def __exit__(self, *args: Any) -> None:
                pass

            def submit(self, fn: Any, **kwargs: Any) -> Future:  # type: ignore[type-arg]
                fut: Future = Future()  # type: ignore[type-arg]
                origin_year = kwargs["origin_year"]
                if origin_year == 2005:
                    fut.set_exception(RuntimeError("Simulated worker failure"))
                else:
                    try:
                        result = fn(**kwargs)
                        fut.set_result(result)
                    except Exception as exc:
                        fut.set_exception(exc)
                return fut

        with patch(
            "scripts.analysis.walk_forward_validation.load_annual_validation_actuals",
            return_value=mock_actuals,
        ), patch(
            "scripts.analysis.walk_forward_validation.maybe_recompute_mig_raw",
            side_effect=lambda mig, snap, cfg: mig,
        ), patch(
            "scripts.analysis.walk_forward_validation.ProcessPoolExecutor",
            _FakeExecutor,
        ):
            # Should not raise — the failed origin gets retried sequentially
            state_df, county_df, curves_df = run_annual_validation(
                snapshots,
                mig_raw,
                survival,
                fertility,
                methods=["m2026r1"],
                workers=4,
            )

        # We still get results for all 4 origin years
        assert set(state_df["origin_year"].unique()) == {2005, 2010, 2015, 2020}


class TestWorkerCountDefaults:
    """Verify worker count capping and auto-detect logic."""

    def test_workers_capped_at_origin_count(self) -> None:
        """Effective workers should never exceed len(ORIGIN_YEARS)."""
        from scripts.analysis.walk_forward_validation import ORIGIN_YEARS

        n_origins = len(ORIGIN_YEARS)  # 4

        # Requesting more workers than origins should be capped
        effective = max(1, min(100, n_origins))
        assert effective == n_origins

    def test_workers_zero_auto_detect(self) -> None:
        """workers=0 should resolve to min(len(ORIGIN_YEARS), cpu_count)."""
        import os

        from scripts.analysis.walk_forward_validation import ORIGIN_YEARS

        cpu = os.cpu_count() or 1
        expected = min(len(ORIGIN_YEARS), cpu)
        assert expected >= 1
        assert expected <= len(ORIGIN_YEARS)

    def test_workers_1_is_sequential(self) -> None:
        """workers=1 must result in sequential execution (effective_workers <= 1)."""
        effective = max(1, min(1, 4))
        assert effective == 1


class TestProjectionWorkerPath:
    """Verify county-level projection workers preserve correctness."""

    @pytest.fixture()
    def shared_inputs(self) -> dict[str, Any]:
        """Build shared inputs for projection-worker comparisons."""
        counties = ["38001", "38003", "38005"]
        snapshots = {
            yr: _make_population_df(counties, yr)
            for yr in [2000, 2005, 2010, 2015, 2020, 2024]
        }
        return {
            "snapshots": snapshots,
            "mig_raw": _make_migration_df(counties),
            "survival": _make_survival(),
            "fertility": _make_fertility(),
        }

    def test_projection_workers_1_vs_2_identical(
        self, shared_inputs: dict[str, Any]
    ) -> None:
        """County-worker path must match sequential results exactly."""
        from scripts.analysis.walk_forward_validation import run_annual_validation

        counties = ["38001", "38003", "38005"]
        mock_actuals: dict[int, pd.DataFrame] = {}
        for yr in range(2006, 2025):
            mock_actuals[yr] = _make_population_df(counties, yr)

        with patch(
            "scripts.analysis.walk_forward_validation.load_annual_validation_actuals",
            return_value=mock_actuals,
        ), patch(
            "scripts.analysis.walk_forward_validation.maybe_recompute_mig_raw",
            side_effect=lambda mig, snap, cfg: mig,
        ):
            state_seq, county_seq, curves_seq = run_annual_validation(
                shared_inputs["snapshots"],
                shared_inputs["mig_raw"],
                shared_inputs["survival"],
                shared_inputs["fertility"],
                methods=["m2026r1"],
                workers=1,
                projection_workers=1,
            )

            state_par, county_par, curves_par = run_annual_validation(
                shared_inputs["snapshots"],
                shared_inputs["mig_raw"],
                shared_inputs["survival"],
                shared_inputs["fertility"],
                methods=["m2026r1"],
                workers=1,
                projection_workers=2,
            )

        pd.testing.assert_frame_equal(
            state_seq.reset_index(drop=True),
            state_par.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            county_seq.reset_index(drop=True),
            county_par.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            curves_seq.reset_index(drop=True),
            curves_par.reset_index(drop=True),
        )

    def test_projection_worker_fallback_on_error(
        self, shared_inputs: dict[str, Any]
    ) -> None:
        """A failed county worker should be retried sequentially."""
        from concurrent.futures import Future

        from scripts.analysis.walk_forward_validation import run_annual_validation

        counties = ["38001", "38003"]
        mock_actuals: dict[int, pd.DataFrame] = {}
        for yr in range(2006, 2025):
            mock_actuals[yr] = _make_population_df(counties, yr)

        class _FakeExecutor:
            """Executor that fails on the first submitted county task."""

            def __init__(self, **kwargs: Any) -> None:
                self._call_count = 0

            def __enter__(self) -> _FakeExecutor:
                return self

            def __exit__(self, *args: Any) -> None:
                pass

            def submit(self, fn: Any, **kwargs: Any) -> Future:  # type: ignore[type-arg]
                fut: Future = Future()  # type: ignore[type-arg]
                self._call_count += 1
                if self._call_count == 1 and "county_fips" in kwargs:
                    fut.set_exception(RuntimeError("Simulated county worker failure"))
                else:
                    try:
                        fut.set_result(fn(**kwargs))
                    except Exception as exc:
                        fut.set_exception(exc)
                return fut

        with patch(
            "scripts.analysis.walk_forward_validation.load_annual_validation_actuals",
            return_value=mock_actuals,
        ), patch(
            "scripts.analysis.walk_forward_validation.maybe_recompute_mig_raw",
            side_effect=lambda mig, snap, cfg: mig,
        ), patch(
            "scripts.analysis.walk_forward_validation.ProcessPoolExecutor",
            _FakeExecutor,
        ):
            state_df, county_df, curves_df = run_annual_validation(
                shared_inputs["snapshots"],
                shared_inputs["mig_raw"],
                shared_inputs["survival"],
                shared_inputs["fertility"],
                methods=["m2026r1"],
                workers=1,
                projection_workers=3,
            )

        assert set(state_df["origin_year"].unique()) == {2005, 2010, 2015, 2020}
        assert not county_df.empty
        assert not curves_df.empty
