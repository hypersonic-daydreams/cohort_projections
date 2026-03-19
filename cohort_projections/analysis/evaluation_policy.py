"""Evaluate benchmark scorecards against the machine-readable evaluation policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
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
    policy.setdefault(
        "operational_quality",
        {
            "require_artifact_completeness": True,
            "require_runtime_summary": True,
            "review_if_reproducibility_logging_missing": True,
            "warn_if_runtime_multiple_of_median_exceeds": 2.5,
            "warn_if_slowest_stage_share_exceeds": 0.75,
        },
    )
    return policy


def _as_bool(value: object) -> bool | None:
    """Return *value* as a boolean when possible."""
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[call-overload]
            return None
    except TypeError:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _as_float(value: object) -> float | None:
    """Return *value* as float when possible."""
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[call-overload]
            return None
    except TypeError:
        pass
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def summarize_runtime_context(
    runtime_history: pd.DataFrame,
    *,
    exclude_run_id: str | None = None,
) -> dict[str, Any]:
    """Summarize archive-relative runtime baselines for operational evaluation."""
    if runtime_history.empty or "total_duration_seconds" not in runtime_history.columns:
        return {}

    frame = runtime_history.copy()
    if exclude_run_id is not None and "run_id" in frame.columns:
        frame = frame[frame["run_id"].astype(str) != str(exclude_run_id)]

    durations = pd.to_numeric(frame["total_duration_seconds"], errors="coerce").dropna()
    durations = durations[durations > 0]
    if durations.empty:
        return {}

    summary: dict[str, Any] = {
        "sample_size": int(len(durations)),
        "median_duration_seconds": float(durations.median()),
        "p90_duration_seconds": float(durations.quantile(0.9)),
    }

    if "slowest_stage" in frame.columns:
        non_empty = frame["slowest_stage"].dropna().astype(str)
        non_empty = non_empty[non_empty.str.strip() != ""]
        if not non_empty.empty:
            summary["slowest_stage_mode"] = str(non_empty.mode().iloc[0])
    return summary


def evaluate_operational_quality(
    challenger_row: dict[str, Any],
    policy: dict[str, Any],
    runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate operational-quality evidence for one challenger row."""
    operational_policy = policy.get("operational_quality", {}) or {}
    runtime_context = runtime_context or {}

    artifact_complete = _as_bool(challenger_row.get("artifact_completeness_flag"))
    runtime_summary_present = _as_bool(challenger_row.get("runtime_summary_present"))
    if runtime_summary_present is None:
        inferred_runtime_summary = any(
            _as_float(challenger_row.get(column)) is not None
            for column in ("runtime_total_seconds", "slowest_stage_seconds", "slowest_stage_share")
        )
        runtime_summary_present = True if inferred_runtime_summary else None
    reproducibility_logging = _as_bool(challenger_row.get("reproducibility_logging_flag"))
    runtime_total_seconds = _as_float(challenger_row.get("runtime_total_seconds"))
    slowest_stage_share = _as_float(challenger_row.get("slowest_stage_share"))
    median_duration_seconds = _as_float(runtime_context.get("median_duration_seconds"))
    runtime_multiple = (
        runtime_total_seconds / median_duration_seconds
        if runtime_total_seconds is not None
        and median_duration_seconds is not None
        and median_duration_seconds > 0
        else None
    )

    results: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    blocked = False
    review_required = False

    require_artifact_completeness = bool(
        operational_policy.get("require_artifact_completeness", True)
    )
    artifact_passed = artifact_complete is not False or not require_artifact_completeness
    results["artifact_completeness"] = {
        "passed": artifact_passed,
        "value": artifact_complete,
        "severity": "blocker",
        "detail": (
            "Benchmark artifacts are complete."
            if artifact_passed
            else "Benchmark artifacts are incomplete, so the run is not reviewable."
        ),
    }
    if not artifact_passed:
        blocked = True
        reasons.append("Operational blocker: benchmark artifacts are incomplete.")

    require_runtime_summary = bool(operational_policy.get("require_runtime_summary", True))
    runtime_summary_required = require_runtime_summary and artifact_complete is not False
    runtime_summary_passed = runtime_summary_present is not False or not runtime_summary_required
    results["runtime_summary_present"] = {
        "passed": runtime_summary_passed,
        "value": runtime_summary_present,
        "severity": "blocker",
        "detail": (
            "Runtime summary evidence is available."
            if runtime_summary_passed
            else "Runtime summary evidence is missing even though the benchmark bundle is otherwise usable."
        ),
    }
    if not runtime_summary_passed:
        blocked = True
        reasons.append("Operational blocker: runtime summary evidence is missing.")

    repro_required = bool(operational_policy.get("review_if_reproducibility_logging_missing", True))
    reproducibility_passed = reproducibility_logging is not False or not repro_required
    results["reproducibility_logging"] = {
        "passed": reproducibility_passed,
        "value": reproducibility_logging,
        "severity": "review",
        "detail": (
            "Reproducibility logging succeeded."
            if reproducibility_passed
            else "Reproducibility logging did not complete; a human should review the run before trusting it."
        ),
    }
    if not reproducibility_passed:
        review_required = True
        reasons.append("Operational review: reproducibility logging was not recorded.")

    runtime_multiple_threshold = _as_float(
        operational_policy.get("warn_if_runtime_multiple_of_median_exceeds")
    )
    runtime_multiple_passed = True
    runtime_multiple_detail = "Runtime is within the archive-relative baseline."
    if runtime_multiple is not None and runtime_multiple_threshold is not None:
        runtime_multiple_passed = runtime_multiple <= runtime_multiple_threshold
        if not runtime_multiple_passed:
            runtime_multiple_detail = (
                f"Runtime is {runtime_multiple:.2f}x the archive median, above the "
                f"{runtime_multiple_threshold:.2f}x review threshold."
            )
    elif runtime_total_seconds is None:
        runtime_multiple_detail = "Runtime duration is not available."
    else:
        runtime_multiple_detail = (
            "Archive runtime baseline is unavailable, so outlier detection was skipped."
        )
    results["runtime_multiple_of_median"] = {
        "passed": runtime_multiple_passed,
        "value": runtime_multiple,
        "threshold": runtime_multiple_threshold,
        "severity": "review",
        "detail": runtime_multiple_detail,
    }
    if not runtime_multiple_passed:
        review_required = True
        reasons.append(runtime_multiple_detail)

    slowest_stage_share_threshold = _as_float(
        operational_policy.get("warn_if_slowest_stage_share_exceeds")
    )
    slowest_stage_passed = True
    slowest_stage_detail = "No single runtime stage dominates the run."
    if slowest_stage_share is not None and slowest_stage_share_threshold is not None:
        slowest_stage_passed = slowest_stage_share <= slowest_stage_share_threshold
        if not slowest_stage_passed:
            slowest_stage_detail = (
                f"The slowest stage consumed {slowest_stage_share:.2f} of total runtime, "
                f"above the {slowest_stage_share_threshold:.2f} review threshold."
            )
    elif slowest_stage_share is None:
        slowest_stage_detail = "Slowest-stage share is not available."
    results["slowest_stage_share"] = {
        "passed": slowest_stage_passed,
        "value": slowest_stage_share,
        "threshold": slowest_stage_share_threshold,
        "severity": "review",
        "detail": slowest_stage_detail,
    }
    if not slowest_stage_passed:
        review_required = True
        reasons.append(slowest_stage_detail)

    return {
        "results": results,
        "blocked": blocked,
        "review_required": review_required,
        "reasons": reasons,
    }


def evaluate_scorecard(
    challenger_row: dict[str, Any],
    champion_row: dict[str, Any],
    policy: dict[str, Any],
    runtime_context: dict[str, Any] | None = None,
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
                f"Hard gate '{gate_name}': PASSED (value={value}, max_allowed={max_allowed})"
            )
        else:
            any_hard_gate_failed = True
            reasons.append(
                f"Hard gate '{gate_name}': FAILED (value={value} > max_allowed={max_allowed})"
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
                f"Tradeoff '{metric}': PASSED (delta={delta:+.4f}, max_regression={max_regression})"
            )
        else:
            any_tradeoff_breached = True
            reasons.append(
                f"Tradeoff '{metric}': BREACHED "
                f"(delta={delta:+.4f} > max_regression={max_regression})"
            )

    # --- Sensitivity flag ---
    sensitivity_flag = bool(challenger_row.get("sensitivity_instability_flag", False))
    if sensitivity_flag:
        reasons.append("Sensitivity instability flag is set on challenger")

    operational = evaluate_operational_quality(challenger_row, policy, runtime_context)

    # --- Classification ---
    if operational["blocked"]:
        classification = "inconclusive"
    elif any_hard_gate_failed:
        classification = "failed_hard_gate"
    elif any_tradeoff_breached or sensitivity_flag or operational["review_required"]:
        classification = "needs_human_review"
    else:
        classification = "passed_all_gates"

    return {
        "classification": classification,
        "hard_gate_results": hard_gate_results,
        "tradeoff_metrics": tradeoff_results,
        "tradeoff_results": tradeoff_results,
        "sensitivity_flag": sensitivity_flag,
        "operational_results": operational["results"],
        "operational_blocked": operational["blocked"],
        "operational_review_required": operational["review_required"],
        "reasons": reasons + operational["reasons"],
    }


def evaluate_promotion(
    challenger_row: dict[str, Any],
    champion_row: dict[str, Any],
    policy: dict[str, Any],
    runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whether a classified challenger is promotion-ready.

    Promotion policy is stricter than queue classification. It requires
    explicit improvements on the recent-state metrics, enforces capped
    regressions on tradeoff metrics, and optionally gives certain sentinel
    counties veto power.
    """
    evaluation = evaluate_scorecard(
        challenger_row,
        champion_row,
        policy,
        runtime_context=runtime_context,
    )
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
        delta = float(challenger_row.get(metric, 0.0)) - float(champion_row.get(metric, 0.0))
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
