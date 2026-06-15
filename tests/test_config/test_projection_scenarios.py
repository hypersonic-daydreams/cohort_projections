"""Tests for production projection scenario configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_projection_config() -> dict:
    """Load the production projection configuration."""
    path = PROJECT_ROOT / "config" / "projection_config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_baseline_uses_cbo_adjusted_assumptions() -> None:
    """The public baseline includes the CBO additive migration and fertility adjustments."""
    config = _load_projection_config()
    baseline = config["scenarios"]["baseline"]

    assert baseline["active"] is True
    assert baseline["fertility"] == "-5_percent"
    assert baseline["migration"]["type"] == "additive_reduction"
    assert baseline["migration"]["schedule"][2025] == 0.20
    assert baseline["migration"]["schedule"][2029] == 0.91
    assert baseline["migration"]["default_factor"] == 1.00
    assert baseline["migration"]["reference_intl_migration"] == 3350.33
    assert baseline["migration"]["reference_population"] == 799358


def test_public_default_run_is_baseline_only() -> None:
    """Only the CBO-adjusted baseline is active by default for PUB-2026 production."""
    config = _load_projection_config()
    active = [key for key, scenario in config["scenarios"].items() if scenario.get("active", False)]

    assert active == ["baseline"]


def test_recent_trend_path_is_retained_as_inactive_sensitivity() -> None:
    """The former unadjusted baseline remains available without driving public runs."""
    config = _load_projection_config()
    recent_trend = config["scenarios"]["recent_trend_continuation"]

    assert recent_trend["active"] is False
    assert recent_trend["fertility"] == "constant"
    assert recent_trend["migration"] == "recent_average"
