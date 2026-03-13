"""Tests for parallel sensitivity-analysis execution."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest


def _make_population_df(counties: list[str], year: int) -> pd.DataFrame:
    """Build a small deterministic population snapshot."""
    age_groups = ["0-4", "5-9"]
    sexes = ["Male", "Female"]
    records: list[dict[str, object]] = []
    for fips in counties:
        seed = int(fips[-3:])
        for age_group in age_groups:
            for sex in sexes:
                records.append(
                    {
                        "county_fips": fips,
                        "age_group": age_group,
                        "sex": sex,
                        "population": 1000.0 + seed * 5 + year * 0.1,
                    }
                )
    return pd.DataFrame(records)


def _make_migration_df(counties: list[str]) -> pd.DataFrame:
    """Build minimal migration rates covering all required periods."""
    periods = [(2000, 2005), (2005, 2010), (2010, 2015), (2015, 2020)]
    age_groups = ["0-4", "5-9"]
    sexes = ["Male", "Female"]
    records: list[dict[str, object]] = []
    for fips in counties:
        for period_start, period_end in periods:
            for age_group in age_groups:
                for sex in sexes:
                    records.append(
                        {
                            "county_fips": fips,
                            "age_group": age_group,
                            "sex": sex,
                            "period_start": period_start,
                            "period_end": period_end,
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
    """Build minimal fertility rates."""
    return {}


@pytest.fixture()
def shared_inputs() -> dict[str, Any]:
    """Shared synthetic inputs for sequential and parallel runs."""
    counties = ["38001", "38003"]
    snapshots = {
        year: _make_population_df(counties, year)
        for year in [2015, 2020, 2024]
    }
    return {
        "snapshots": snapshots,
        "mig_raw": _make_migration_df(counties),
        "survival": _make_survival(),
        "fertility": _make_fertility(),
    }


class TestParallelSensitivityIdenticalResults:
    """Verify sequential and parallel sensitivity runs match exactly."""

    def test_workers_1_vs_2_identical(self, shared_inputs: dict[str, Any]) -> None:
        """Parallel workers should preserve deterministic results."""
        from scripts.analysis import sensitivity_analysis as sa

        small_perturbations = {
            "migration_rate": [
                {"label": "baseline", "factor": 1.0},
                {"label": "+10%", "factor": 1.1},
            ]
        }

        with patch.object(sa, "PERTURBATIONS", small_perturbations):
            seq = sa.run_sensitivity_analysis(
                shared_inputs["snapshots"],
                shared_inputs["mig_raw"],
                shared_inputs["survival"],
                shared_inputs["fertility"],
                methods=["m2026r1"],
                workers=1,
            )
            par = sa.run_sensitivity_analysis(
                shared_inputs["snapshots"],
                shared_inputs["mig_raw"],
                shared_inputs["survival"],
                shared_inputs["fertility"],
                methods=["m2026r1"],
                workers=2,
            )

        pd.testing.assert_frame_equal(
            seq.reset_index(drop=True),
            par.reset_index(drop=True),
        )


class TestParallelSensitivityFallback:
    """Verify failed worker tasks fall back to sequential execution."""

    def test_fallback_on_worker_error(self, shared_inputs: dict[str, Any]) -> None:
        """A failed worker task should be retried sequentially."""
        from concurrent.futures import Future

        from scripts.analysis import sensitivity_analysis as sa

        small_perturbations = {
            "migration_rate": [
                {"label": "baseline", "factor": 1.0},
                {"label": "+10%", "factor": 1.1},
            ]
        }

        class _FakeExecutor:
            """Executor that fails on the first submitted task only."""

            def __init__(self, **kwargs: Any) -> None:
                self._call_count = 0

            def __enter__(self) -> _FakeExecutor:
                return self

            def __exit__(self, *args: Any) -> None:
                pass

            def submit(self, fn: Any, **kwargs: Any) -> Future:  # type: ignore[type-arg]
                fut: Future = Future()  # type: ignore[type-arg]
                self._call_count += 1
                if self._call_count == 1:
                    fut.set_exception(RuntimeError("Simulated worker failure"))
                else:
                    try:
                        fut.set_result(fn(**kwargs))
                    except Exception as exc:
                        fut.set_exception(exc)
                return fut

        with patch.object(sa, "PERTURBATIONS", small_perturbations), patch.object(
            sa, "ProcessPoolExecutor", _FakeExecutor
        ):
            results = sa.run_sensitivity_analysis(
                shared_inputs["snapshots"],
                shared_inputs["mig_raw"],
                shared_inputs["survival"],
                shared_inputs["fertility"],
                methods=["m2026r1"],
                workers=4,
            )

        assert len(results) == 4
        assert set(results["perturbation_level"]) == {"baseline", "+10%"}
        assert set(results["origin_year"]) == {2015, 2020}
