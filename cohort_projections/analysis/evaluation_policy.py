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
        - tradeoff_results: dict mapping metric to
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
        "tradeoff_results": tradeoff_results,
        "sensitivity_flag": sensitivity_flag,
        "reasons": reasons,
    }
