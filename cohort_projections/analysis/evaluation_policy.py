"""Evaluate benchmark scorecards against the machine-readable evaluation policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "config" / "benchmark_evaluation_policy.yaml"

# Valid classification outcomes
CLASSIFICATIONS = frozenset(
    {
        "passed_all_gates",
        "needs_human_review",
        "failed_hard_gate",
        "inconclusive",
    }
)


def load_policy(policy_path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    """Load the evaluation policy from YAML."""
    if not policy_path.exists():
        raise FileNotFoundError(f"Evaluation policy not found: {policy_path}")
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise ValueError(f"Invalid evaluation policy format: {policy_path}")
    # Validate required top-level keys
    if "tradeoff_thresholds" not in policy and "tradeoff_metrics" in policy:
        policy["tradeoff_thresholds"] = policy["tradeoff_metrics"]
    if "tradeoff_metrics" not in policy and "tradeoff_thresholds" in policy:
        policy["tradeoff_metrics"] = policy["tradeoff_thresholds"]

    for key in ("hard_gates", "tradeoff_thresholds", "agent_classification_rules"):
        if key not in policy:
            raise ValueError(f"Policy missing required key: {key}")
    return policy


def evaluate_scorecard(
    challenger_row: dict[str, Any],
    champion_row: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a challenger scorecard row against the champion using the policy.

    Parameters
    ----------
    challenger_row : dict
        A single row from summary_scorecard.csv for the challenger method.
        Expected keys include all scorecard columns from build_summary_scorecard().
    champion_row : dict
        A single row from summary_scorecard.csv for the champion method.
    policy : dict
        Loaded evaluation policy from load_policy().

    Returns
    -------
    dict with keys:
        - classification: one of CLASSIFICATIONS
        - hard_gate_results: dict mapping gate name to
          {passed: bool, value: int/float, max_allowed: int/float}
        - tradeoff_metrics: dict mapping metric to
          {passed: bool, delta: float, max_regression: float}
        - sensitivity_flag: bool
        - reasons: list of strings explaining the classification
    """
    hard_gate_results: dict[str, dict[str, Any]] = {}
    tradeoff_results: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []

    # --- Hard gate evaluation ---
    any_hard_gate_failed = False
    for gate_name, gate_spec in policy["hard_gates"].items():
        max_allowed = gate_spec["max_allowed"]
        value = challenger_row.get(gate_name, 0)
        passed = value <= max_allowed
        hard_gate_results[gate_name] = {
            "passed": passed,
            "value": value,
            "max_allowed": max_allowed,
        }
        if passed:
            reasons.append(
                f"Hard gate '{gate_name}': PASSED "
                f"(value={value}, max_allowed={max_allowed})"
            )
        else:
            any_hard_gate_failed = True
            reasons.append(
                f"Hard gate '{gate_name}': FAILED "
                f"(value={value} > max_allowed={max_allowed})"
            )

    # --- Tradeoff threshold evaluation ---
    any_tradeoff_breached = False
    for metric, threshold_spec in policy["tradeoff_thresholds"].items():
        max_regression = threshold_spec["max_regression"]
        challenger_val = challenger_row.get(metric, 0.0)
        champion_val = champion_row.get(metric, 0.0)
        delta = challenger_val - champion_val
        passed = delta <= max_regression
        tradeoff_results[metric] = {
            "passed": passed,
            "delta": round(delta, 6),
            "max_regression": max_regression,
        }
        if passed:
            reasons.append(
                f"Tradeoff '{metric}': PASSED "
                f"(delta={delta:+.4f}, max_regression={max_regression})"
            )
        else:
            any_tradeoff_breached = True
            reasons.append(
                f"Tradeoff '{metric}': BREACHED "
                f"(delta={delta:+.4f} > max_regression={max_regression})"
            )

    # --- Sensitivity flag ---
    sensitivity_flag = bool(
        challenger_row.get("sensitivity_instability_flag", False)
    )
    if sensitivity_flag:
        reasons.append("Sensitivity instability flag is set on challenger")

    # --- Classification ---
    if any_hard_gate_failed:
        classification = "failed_hard_gate"
    elif any_tradeoff_breached or sensitivity_flag:
        classification = "needs_human_review"
    else:
        classification = "passed_all_gates"

    return {
        "classification": classification,
        "hard_gate_results": hard_gate_results,
        "tradeoff_metrics": tradeoff_results,
        "tradeoff_results": tradeoff_results,
        "sensitivity_flag": sensitivity_flag,
        "reasons": reasons,
    }


def evaluate_promotion(
    challenger_row: dict[str, Any],
    champion_row: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate whether a classified challenger is promotion-ready.

    Promotion policy is stricter than queue classification. It requires
    explicit improvements on the recent-state metrics, enforces capped
    regressions on tradeoff metrics, and optionally gives certain sentinel
    counties veto power.
    """
    evaluation = evaluate_scorecard(challenger_row, champion_row, policy)
    promotion_policy = policy.get("promotion_policy", {})
    reasons: list[str] = []
    checks: dict[str, Any] = {"base_classification": evaluation["classification"]}

    if evaluation["classification"] != "passed_all_gates":
        reasons.append("Challenger did not pass the baseline gate-aware classification.")
        return {"promotion_ready": False, "checks": checks, "reasons": reasons}

    required_improvements = {
        "state_ape_recent_short": promotion_policy.get(
            "state_ape_recent_short_min_improvement", 0.0
        ),
        "state_ape_recent_medium": promotion_policy.get(
            "state_ape_recent_medium_min_improvement", 0.0
        ),
    }
    for metric, min_improvement in required_improvements.items():
        challenger_value = float(challenger_row.get(metric, 0.0))
        champion_value = float(champion_row.get(metric, 0.0))
        improvement = champion_value - challenger_value
        passed = improvement >= float(min_improvement)
        checks[metric] = {
            "passed": passed,
            "improvement": round(improvement, 6),
            "min_improvement": float(min_improvement),
        }
        if not passed:
            reasons.append(
                f"{metric} improvement {improvement:+.4f} is below the required "
                f"{float(min_improvement):.4f}."
            )

    overall_regression_limit = promotion_policy.get("county_mape_overall_max_regression")
    if overall_regression_limit is not None:
        delta = float(challenger_row.get("county_mape_overall", 0.0)) - float(
            champion_row.get("county_mape_overall", 0.0)
        )
        passed = delta <= float(overall_regression_limit)
        checks["county_mape_overall"] = {
            "passed": passed,
            "delta": round(delta, 6),
            "max_regression": float(overall_regression_limit),
        }
        if not passed:
            reasons.append(
                f"county_mape_overall regression {delta:+.4f} exceeds "
                f"{float(overall_regression_limit):.4f}."
            )

    sentinel_vetoes = policy.get("promotion_policy", {}).get("sentinel_vetoes", {})
    for metric, max_regression in sentinel_vetoes.items():
        delta = float(challenger_row.get(metric, 0.0)) - float(
            champion_row.get(metric, 0.0)
        )
        passed = delta <= float(max_regression)
        checks[metric] = {
            "passed": passed,
            "delta": round(delta, 6),
            "max_regression": float(max_regression),
        }
        if not passed:
            reasons.append(
                f"{metric} regression {delta:+.4f} triggered a sentinel veto "
                f"(limit {float(max_regression):.4f})."
            )

    return {
        "promotion_ready": not reasons,
        "checks": checks,
        "reasons": reasons,
        "classification": evaluation["classification"],
    }
