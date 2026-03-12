"""Tests for cohort_projections.analysis.evaluation_policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from cohort_projections.analysis.evaluation_policy import (
    DEFAULT_POLICY_PATH,
    evaluate_scorecard,
    load_policy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_POLICY: dict[str, Any] = {
    "hard_gates": {
        "negative_population_violations": {"max_allowed": 0},
        "scenario_order_violations": {"max_allowed": 0},
        "aggregation_violations": {"max_allowed": 0},
    },
    "tradeoff_thresholds": {
        "county_mape_rural": {"max_regression": 0.10},
        "county_mape_bakken": {"max_regression": 0.50},
        "county_mape_overall": {"max_regression": 0.05},
    },
    "agent_classification_rules": {},
}


def _make_row(
    *,
    negative_population_violations: int = 0,
    scenario_order_violations: int = 0,
    aggregation_violations: int = 0,
    county_mape_rural: float = 0.05,
    county_mape_bakken: float = 0.20,
    county_mape_overall: float = 0.03,
    sensitivity_instability_flag: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Build a synthetic scorecard row with sensible defaults."""
    row: dict[str, Any] = {
        "negative_population_violations": negative_population_violations,
        "scenario_order_violations": scenario_order_violations,
        "aggregation_violations": aggregation_violations,
        "county_mape_rural": county_mape_rural,
        "county_mape_bakken": county_mape_bakken,
        "county_mape_overall": county_mape_overall,
        "sensitivity_instability_flag": sensitivity_instability_flag,
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# load_policy tests
# ---------------------------------------------------------------------------


class TestLoadPolicy:
    """Tests for load_policy()."""

    def test_load_policy_default(self) -> None:
        """Load the real policy YAML and verify required keys are present."""
        policy = load_policy(DEFAULT_POLICY_PATH)
        assert isinstance(policy, dict)
        for key in ("hard_gates", "tradeoff_thresholds", "agent_classification_rules"):
            assert key in policy, f"Missing required key: {key}"
        # Sanity-check a known gate
        assert "negative_population_violations" in policy["hard_gates"]

    def test_load_policy_missing_file(self, tmp_path: Path) -> None:
        """FileNotFoundError when the path does not exist."""
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_policy(missing)

    def test_load_policy_invalid_format(self, tmp_path: Path) -> None:
        """ValueError when the YAML content is not a dict."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid evaluation policy format"):
            load_policy(bad_file)

    def test_load_policy_missing_required_key(self, tmp_path: Path) -> None:
        """ValueError when a required top-level key is absent."""
        incomplete = {
            "hard_gates": {},
            "tradeoff_thresholds": {},
            # agent_classification_rules intentionally missing
        }
        policy_file = tmp_path / "incomplete.yaml"
        policy_file.write_text(yaml.dump(incomplete), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required key"):
            load_policy(policy_file)


# ---------------------------------------------------------------------------
# evaluate_scorecard tests
# ---------------------------------------------------------------------------


class TestEvaluateScorecard:
    """Tests for evaluate_scorecard()."""

    @pytest.fixture()
    def policy(self) -> dict[str, Any]:
        return SAMPLE_POLICY

    def test_evaluate_passed_all_gates(self, policy: dict[str, Any]) -> None:
        """Challenger is better on everything → passed_all_gates."""
        champion = _make_row(county_mape_rural=0.10, county_mape_overall=0.05)
        challenger = _make_row(county_mape_rural=0.08, county_mape_overall=0.03)

        result = evaluate_scorecard(challenger, champion, policy)
        assert result["classification"] == "passed_all_gates"
        assert result["sensitivity_flag"] is False
        assert all(g["passed"] for g in result["hard_gate_results"].values())
        assert all(t["passed"] for t in result["tradeoff_results"].values())

    def test_evaluate_failed_hard_gate_negative_pop(
        self, policy: dict[str, Any]
    ) -> None:
        """Challenger has negative_population_violations=1 → failed_hard_gate."""
        champion = _make_row()
        challenger = _make_row(negative_population_violations=1)

        result = evaluate_scorecard(challenger, champion, policy)
        assert result["classification"] == "failed_hard_gate"
        gate = result["hard_gate_results"]["negative_population_violations"]
        assert gate["passed"] is False
        assert gate["value"] == 1

    def test_evaluate_failed_hard_gate_aggregation(
        self, policy: dict[str, Any]
    ) -> None:
        """Challenger has aggregation_violations=2 → failed_hard_gate."""
        champion = _make_row()
        challenger = _make_row(aggregation_violations=2)

        result = evaluate_scorecard(challenger, champion, policy)
        assert result["classification"] == "failed_hard_gate"
        gate = result["hard_gate_results"]["aggregation_violations"]
        assert gate["passed"] is False
        assert gate["value"] == 2

    def test_evaluate_failed_hard_gate_scenario_order(
        self, policy: dict[str, Any]
    ) -> None:
        """Challenger has scenario_order_violations=1 → failed_hard_gate."""
        champion = _make_row()
        challenger = _make_row(scenario_order_violations=1)

        result = evaluate_scorecard(challenger, champion, policy)
        assert result["classification"] == "failed_hard_gate"
        gate = result["hard_gate_results"]["scenario_order_violations"]
        assert gate["passed"] is False
        assert gate["value"] == 1

    def test_evaluate_needs_review_tradeoff_breach(
        self, policy: dict[str, Any]
    ) -> None:
        """Hard gates pass but county_mape_rural delta > 0.10 → needs_human_review."""
        champion = _make_row(county_mape_rural=0.10)
        # challenger is 0.21 higher than champion → delta = 0.11 > 0.10
        challenger = _make_row(county_mape_rural=0.21)

        result = evaluate_scorecard(challenger, champion, policy)
        assert result["classification"] == "needs_human_review"
        assert result["tradeoff_results"]["county_mape_rural"]["passed"] is False

    def test_evaluate_needs_review_sensitivity_flag(
        self, policy: dict[str, Any]
    ) -> None:
        """Hard gates and tradeoffs pass but sensitivity flag → needs_human_review."""
        champion = _make_row()
        challenger = _make_row(sensitivity_instability_flag=True)

        result = evaluate_scorecard(challenger, champion, policy)
        assert result["classification"] == "needs_human_review"
        assert result["sensitivity_flag"] is True
        # All gates and tradeoffs should still pass
        assert all(g["passed"] for g in result["hard_gate_results"].values())
        assert all(t["passed"] for t in result["tradeoff_results"].values())

    def test_evaluate_exactly_at_threshold(self, policy: dict[str, Any]) -> None:
        """Delta exactly equals max_regression → should pass (check is <=)."""
        champion = _make_row(county_mape_rural=0.10)
        # delta = 0.20 - 0.10 = 0.10, exactly at max_regression
        challenger = _make_row(county_mape_rural=0.20)

        result = evaluate_scorecard(challenger, champion, policy)
        assert result["classification"] == "passed_all_gates"
        trade = result["tradeoff_results"]["county_mape_rural"]
        assert trade["passed"] is True
        assert trade["delta"] == pytest.approx(0.10)

    def test_evaluate_multiple_violations(self, policy: dict[str, Any]) -> None:
        """Both hard gate failure and tradeoff breach → failed_hard_gate (priority)."""
        champion = _make_row(county_mape_rural=0.10)
        challenger = _make_row(
            negative_population_violations=1,
            county_mape_rural=0.30,  # delta 0.20 > 0.10
        )

        result = evaluate_scorecard(challenger, champion, policy)
        assert result["classification"] == "failed_hard_gate"
        assert result["hard_gate_results"]["negative_population_violations"]["passed"] is False
        assert result["tradeoff_results"]["county_mape_rural"]["passed"] is False

    def test_evaluate_missing_fields_default_to_zero(
        self, policy: dict[str, Any]
    ) -> None:
        """Challenger row missing gate fields should default to 0 and pass."""
        champion: dict[str, Any] = {}  # empty — defaults to 0.0 for tradeoffs too
        challenger: dict[str, Any] = {}  # empty — defaults to 0 for gates

        result = evaluate_scorecard(challenger, champion, policy)
        assert result["classification"] == "passed_all_gates"
        # All hard gates should show value=0
        for gate in result["hard_gate_results"].values():
            assert gate["value"] == 0
            assert gate["passed"] is True
        # All tradeoff deltas should be 0
        for trade in result["tradeoff_results"].values():
            assert trade["delta"] == pytest.approx(0.0)
            assert trade["passed"] is True
        assert result["sensitivity_flag"] is False
