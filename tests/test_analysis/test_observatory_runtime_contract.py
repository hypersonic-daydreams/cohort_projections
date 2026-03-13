"""Tests for Observatory runtime contract introspection."""

from __future__ import annotations

from cohort_projections.analysis.observatory.runtime_contract import (
    get_runtime_injectable_parameters,
    is_runtime_injectable,
)


def test_runtime_contract_includes_known_injectable_parameters() -> None:
    params = get_runtime_injectable_parameters()
    assert "college_blend_factor" in params
    assert "rate_cap_general" in params
    assert "boom_period_dampening" in params


def test_runtime_contract_excludes_noninjectable_parameter() -> None:
    assert is_runtime_injectable("mortality_improvement_factor") is False
