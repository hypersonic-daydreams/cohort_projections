"""
Integration tests for place projection pipeline stage (PP-003 IMP-11).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

stage_mod = importlib.import_module("scripts.pipeline.02a_run_place_projections")


def _build_config(projection_root: Path) -> dict[str, Any]:
    """Build minimal config for stage tests."""
    return {
        "pipeline": {"projection": {"output_dir": str(projection_root)}},
        "place_projections": {"enabled": True},
        "scenarios": {
            "baseline": {"active": True},
            "restricted_growth": {"active": True},
            "high_growth": {"active": True},
            "zero_migration": {"active": False},
        },
    }


def _write_winner(root: Path) -> Path:
    """Write winner payload at default stage location."""
    winner_path = root / "data" / "backtesting" / "place_backtest_results" / "backtest_winner.json"
    winner_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "winner_variant_id": "B-II",
        "fitting_method": "wls",
        "constraint_method": "cap_and_redistribute",
        "acceptance": {"all_scored_tiers_pass_primary": True},
    }
    winner_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return winner_path


def test_main_dry_run_validates_dependencies_without_running_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Dry-run validates winner + county inputs and does not call orchestrator."""
    projection_root = tmp_path / "projections"
    config = _build_config(projection_root)

    for scenario in ["baseline", "restricted_growth", "high_growth"]:
        (projection_root / scenario / "county").mkdir(parents=True, exist_ok=True)

    _write_winner(tmp_path)

    monkeypatch.setattr(stage_mod, "project_root", tmp_path)
    monkeypatch.setattr(stage_mod, "load_projection_config", lambda _: config)

    def _unexpected_call(**_: Any) -> dict[str, Any]:
        raise AssertionError("run_place_projections should not be called during --dry-run")

    monkeypatch.setattr(stage_mod, "run_place_projections", _unexpected_call)
    monkeypatch.setattr(
        sys,
        "argv",
        ["02a_run_place_projections.py", "--dry-run"],
    )

    assert stage_mod.main() == 0


def test_main_runs_all_active_scenarios_and_writes_place_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stage runs all active scenarios and creates place output directories."""
    projection_root = tmp_path / "projections"
    config = _build_config(projection_root)
    _write_winner(tmp_path)

    called: list[str] = []

    def _fake_run_place_projections(
        scenario: str,
        config: dict[str, Any],
        variant_winner: dict[str, Any] | str,
    ) -> dict[str, Any]:
        called.append(scenario)
        assert isinstance(variant_winner, dict)
        output_dir = Path(config["pipeline"]["projection"]["output_dir"]) / scenario / "place"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "places_summary.csv").write_text("place_fips\n", encoding="utf-8")
        return {
            "scenario": scenario,
            "places_processed": 0,
            "balance_rows": 0,
            "summary_path": output_dir / "places_summary.csv",
        }

    monkeypatch.setattr(stage_mod, "project_root", tmp_path)
    monkeypatch.setattr(stage_mod, "load_projection_config", lambda _: config)
    monkeypatch.setattr(stage_mod, "run_place_projections", _fake_run_place_projections)
    monkeypatch.setattr(sys, "argv", ["02a_run_place_projections.py"])

    assert stage_mod.main() == 0
    assert called == ["baseline", "restricted_growth", "high_growth"]

    for scenario in called:
        place_dir = projection_root / scenario / "place"
        assert place_dir.exists()
        assert (place_dir / "places_summary.csv").exists()


def test_main_honors_explicit_scenario_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit --scenarios list overrides active scenario set."""
    projection_root = tmp_path / "projections"
    config = _build_config(projection_root)
    _write_winner(tmp_path)

    called: list[str] = []

    def _fake_run_place_projections(
        scenario: str,
        config: dict[str, Any],
        variant_winner: dict[str, Any] | str,
    ) -> dict[str, Any]:
        called.append(scenario)
        output_dir = Path(config["pipeline"]["projection"]["output_dir"]) / scenario / "place"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "places_summary.csv").write_text("place_fips\n", encoding="utf-8")
        return {
            "scenario": scenario,
            "places_processed": 0,
            "balance_rows": 0,
            "summary_path": output_dir / "places_summary.csv",
        }

    monkeypatch.setattr(stage_mod, "project_root", tmp_path)
    monkeypatch.setattr(stage_mod, "load_projection_config", lambda _: config)
    monkeypatch.setattr(stage_mod, "run_place_projections", _fake_run_place_projections)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "02a_run_place_projections.py",
            "--scenarios",
            "baseline",
            "high_growth",
        ],
    )

    assert stage_mod.main() == 0
    assert called == ["baseline", "high_growth"]
